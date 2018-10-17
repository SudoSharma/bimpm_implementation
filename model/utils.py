from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from abc import ABC, abstractmethod


class ModelData(ABC):
    @abstractmethod
    def __init__(self, args):
        pass

    def build_char_vocab(self):
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.word_chars.append(chars)

    def words_to_chars(self, batch):
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return = [[self.word_chars]]


class SNLI(ModelData):
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, tokenize=)
        self.LABEL = data.LabelField()

        self.train, self.valid, self.test = data.SNLI.splits(
                self.TEXT, self.LABEL)

        self.TEXT.build_vocab(
                self.train, self.valid, self.test,
                vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.valid_iter = data.BucketIterator.splits(
                (self.train, self.valid, self.test),
                batch_sizes=[args.batch_size]*3,
                device=args.gpu)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        self.char_vocab = {'': 0}
        self.word_chars = [[0] * self.max_word_len, [0] * self.max_word_len]

        self.build_char_vocab()


class Quora(ModelData):
    def __init__(self, args):
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True)
        self.LABEL = data.LabelField()

        self.fields = [
                ('label', self.LABEL),
                ('q1', self.TEXT),
                ('q2', self.TEXT),
                ('q2', self.RAW)]

        self.train, self.valid, self.test = data.TabularDataset.splits(
                path='./data/quora',
                train='train.tsv',
                validation='dev.tsv',
                test='test.tsv',
                format='tsv',
                fields=self.fields)

        self.TEXT.build_vocab(
                self.train, self.valid, self.test,
                vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.valid_iter = data.BucketIterator.splits(
                (self.train, self.valid, self.test),
                batch_sizes=[args.batch_size]*3,
                device=args.gpu,
                sort_key=self.sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        self.char_vocab = {'': 0}
        self.word_chars = [[0] * self.max_word_len, [0] * self.max_word_len]

        self.build_char_vocab()

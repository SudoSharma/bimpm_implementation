import torch
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from abc import ABC, abstractmethod


class SNLI:
    def __init__(self, args):
        self.args = args

        self.TEXT = data.Field(batch_first=True, tokenize='spacy')
        self.LABEL = data.LabelField()

        self.train, self.valid, self.test = datasets.SNLI.splits(
            self.TEXT, self.LABEL)

        self.TEXT.build_vocab(
            self.train,
            self.valid,
            self.test,
            vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.valid_iter, self.test_iter = \
            data.BucketIterator.splits(
                (self.train, self.valid, self.test),
                batch_sizes=[args.batch_size] * 3,
                device=torch.device(args.device))

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        self.char_vocab = {'': 0}
        self.word_chars = [[0] * self.max_word_len, [0] * self.max_word_len]

        self.build_char_vocab()

        self.last_epoch = -1

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
        return [[self.word_chars[w] for w in words] for words in batch]

    def keep_training(self, iterator):
        self.present_epoch = int(iterator.epoch)
        if self.present_epoch == self.args.epoch:
            return False
        if self.present_epoch > self.last_epoch:
            print(f'epoch: {self.present_epoch+1}')
        self.last_epoch = self.present_epoch
        return True


class Quora:
    def __init__(self, args, toy=False):
        self.args = args

        self.RAW = data.RawField()
        self.RAW.is_target = False
        self.TEXT = data.Field(batch_first=True)
        self.LABEL = data.LabelField()

        self.fields = [('label', self.LABEL), ('q1', self.TEXT),
                       ('q2', self.TEXT), ('id', self.RAW)]

        self.train, self.valid, self.test = data.TabularDataset.splits(
            path='./data/quora',
            train='toy_train.tsv' if toy else 'train.tsv',
            validation='toy_dev.tsv' if toy else 'dev.tsv',
            test='toy_test.tsv' if toy else 'test.tsv',
            format='tsv',
            fields=self.fields)

        self.TEXT.build_vocab(
            self.train,
            self.valid,
            self.test,
            vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.valid_iter, self.test_iter = \
            data.BucketIterator.splits(
                (self.train, self.valid, self.test),
                batch_sizes=[args.batch_size] * 3,
                device=torch.device(args.device),
                sort_key=self.sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        self.char_vocab = {'': 0}
        self.word_chars = [[0] * self.max_word_len, [0] * self.max_word_len]

        self.build_char_vocab()

        self.last_epoch = -1

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
        return [[self.word_chars[w] for w in words] for words in batch]

    def keep_training(self, iterator):
        self.present_epoch = int(iterator.epoch)
        if self.present_epoch == self.args.epoch:
            return False
        if self.present_epoch > self.last_epoch:
            print(f'epoch: {self.present_epoch+1}')
        self.last_epoch = self.present_epoch
        return True


class Sentence:
    def __init__(self, batch, model_data, data_type):
        self.batch, self.model_data = batch, model_data

        if data_type == 'SNLI':
            self.p, self.q = 'premise', 'hypothesis'
        else:
            self.p, self.q = 'q1', 'q2'

    def process_batch(self, device):
        self.p = getattr(self.batch, self.p)
        self.q = getattr(self.batch, self.q)

        self.char_p = Variable(
            torch.LongTensor(self.model_data.words_to_chars(self.p)))
        self.char_q = Variable(
            torch.LongTensor(self.model_data.words_to_chars(self.q)))

        self.char_p.to(device)
        self.char_q.to(device)

    def make_data_dict(self):
        self.p = {'words': self.p, 'chars': self.char_p}
        self.q = {'words': self.q, 'chars': self.char_q}

    def generate(self, gpu):
        self.process_batch(gpu)
        self.make_data_dict()
        return (self.p, self.q)


class Args:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)

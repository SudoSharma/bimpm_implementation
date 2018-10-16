from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


class Quora:
    def __init__(self, args):
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first)
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

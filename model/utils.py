from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


class Quora:
    def __init__(self, args):
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first)
        self.LABEL = data.LabelField()

        quora_fields = [
                ('label', self.LABEL),
                ('q1', self.TEXT),
                ('q2', self.TEXT)]
                ('q2', self.TEXT


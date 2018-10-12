import torch
import torch.nn as nn
import plac


def main():
    pass


class CharacterRepresentationLayer(nn.Module):
    def __init__(self):
        super(CharacterRepresentationLayer, self).__init__()

        self.lstm = nn.LSTM()

    def forward(self):
        pass


class WordRepresentationLayer(nn.Module):
    def __init__(self):
        super(WordRepresentationLayer, self).__init__()
        pass

    def forward(self):
        pass


class ContextRepresentationLayer(nn.Module):
    def __init__(self):
        super(ContextRepresentationLayer, self).__init__()
        pass

    def forward(self):
        pass


class MatchingLayer(nn.Module):
    def __init__(self):
        super(MatchingLayer, self).__init__()
        pass

    def forward(self):
        pass


class AggregationLayer(nn.Module):
    def __init__(self):
        super(AggregationLayer, self).__init__()
        pass

    def forward(self):
        pass


class PredictionLayer(nn.Module):
    def __init__(self):
        super(PredictionLayer, self).__init__()
        pass

    def forward(self):
        pass


if __name__() == "__main__":
    plac.call(main)

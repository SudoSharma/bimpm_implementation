import torch
import torch.nn as nn
import model.layer as L
import plac


def main():
    pass


class BiMPM(nn.Module):
    def __init__(self, args, data):
        super(BiMPM, self).__init__()

        self.args = args
        self.w_layer = L.WordRepresentationLayer(args, data)
        self.c_layer = L.ContextRepresentationLayer(args)
        self.m_layer = L.MatchingLayer(args)
        self.a_layer = L.AggregationLayer(args)
        self.p_layer = L.PredictionLayer(args)


    def forward(self, p, q):
        # TODO create sentence object
        # 'p' should be a sentence object with two attributes,
        # a chars attribute of shape (seq_len, max_word_len) 
        # a words attribute of shape (batch_size, seq_len)
        p, q = self.w_layer(p), self.w_layer(q)
        p, q = self.c_layer(p), self.c_layer(q)
        p, q = self.m_layer(p, q)
        match_vec = self.a_layer(p, q)

        return self.p_layer(match_vec)


if __name__() == "__main__":
    plac.call(main)

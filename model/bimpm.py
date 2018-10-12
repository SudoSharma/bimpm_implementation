import torch
import torch.nn as nn
import model.layer as layer
import plac


def main():
    pass


class BiMPM(nn.Module):
    def __init__(self):

        super(BiMPM, self).__init__()

        self.w_layer = layer.WordRepresentationLayer()
        self.c_layer = layer.ContextRepresentationLayer()
        self.m_layer = layer.MatchingLayer()
        self.a_layer = layer.AggregationLayer()
        self.p_layer = layer.PredictionLayer()


    def forward(self, p, q):

        p, q = self.w_layer(p), self.w_layer(q)
        p, q = self.c_layer(p), self.c_layer(q)
        p, q = self.m_layer(p, q)
        p, q = self.a_layer(p), self.a_layer(q)

        return self.p_layer(p, q)


if __name__() == "__main__":
    plac.call(main)

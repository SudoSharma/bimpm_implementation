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
        self.m_layer = MatchingLayer()
        self.a_layer = AggregationLayer()
        self.p_layer = PredictionLayer()


    def forward(self, data):
        p, q = data
        p_out, q_out = self.w_layer(p, q)

        pass


if __name__() == "__main__":
    plac.call(main)

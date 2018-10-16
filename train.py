import torch
from torch import nn, optim
from torch.autograd import Variable

from model.bimpm import BiMPM
import plac


def main():
    pass


def train(args, data)
   model = BiMPM(args, data)
   parameters = (p for p in model.parameters() if p.requires_grad)
   optimizer = optim.Adam(parameters, lr=args.lr)
   criterion = nn.CrossEntropyLoss()

   model.train()
   


if __name__() == '__main__':
    plac.call(main)

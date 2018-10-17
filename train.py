import copy
import torch
from torch import nn, optim
from torch.autograd import Variable

from test import test
from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence
import plac


def main():
    pass


def train(args, model_data):
    model = BiMPM(args, model_data)
    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss = 0
    max_valid_acc, max_test_acc = 0, 0

    iterator = model_data.train_iter
    for i, batch in enumerate(iterator):
        if not model_data.keep_training(iterator):
            break
        p, q = Sentence(batch, model_data, args.data_type).generate()

        preds = model(p, q)

        optimizer.zero_grad()
        batch_loss = criterion(preds, batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_interval == 0:
            valid_loss, valid_acc = test(model, args, model_data, mode='valid')
            test_loss, test_acc = test(model, args, model_data)
            c = (i + 1) // args.print_interval

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    return best_model


if __name__() == '__main__':
    plac.call(main)

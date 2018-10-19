import os
import copy
from time import gmtime, strftime
import plac

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence, Args
from test import test


def main(batch_size: ('[64]', 'positional', None, int)=64,
         char_input_size: ('[20]', 'positional', None, int)=20,
         char_hidden_size: ('[50]', 'positional', None, int)=50,
         data_type: ("{SNLI, [Quora]}") = 'quora',
         dropout: ('[0.1]', 'positional', None, float)=0.1,
         epoch: ('[10]', 'positional', None, int)=10,
         hidden_size: ('[100]', 'positional', None, int)=100,
         lr: ('[0.001]', 'positional', None, float)=0.001,
         num_perspectives: ('[20]', 'positional', None, int)=20,
         print_interval: ('[500]', 'positional', None, int)=500,
         word_dim: ('[300]', 'positional', None, int)=300):
    args = Args(locals())

    args.device = torch.device('cuda:0' if torch.cuda.
                               is_available() else 'cpu')

    if args.data_type.lower() == 'snli':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type.lower() == 'quora':
        print("Loading Quora data...")
        model_data = Quora(args, toy=True)
        # model_data = Quora(args)
    else:
        raise RuntimeError(
            'Data source other than SNLI or Quora was provided.')

    args.char_vocab_size = len(model_data.char_vocab)
    args.word_vocab_size = len(model_data.TEXT.vocab)
    args.class_size = len(model_data.LABEL.vocab)
    args.max_word_len = model_data.max_word_len
    args.model_time = strftime('%H:%M:%S', gmtime())

    print("Starting training...")
    best_model = train(args, model_data)

    if not os.path.exists('saved_models'):
        os.makedirs('save_models')
    torch.save(best_model.state_dict(),
               f'saved_models/bimpm_{args.data_type}_{args.model_time}.pt')

    print("Finished training...")


def train(args, model_data):
    model = BiMPM(args, model_data)
    model.to(args.device)

    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    train_loss, max_valid_acc, max_test_acc = 0, 0, 0

    iterator = model_data.train_iter
    for i, batch in enumerate(iterator):
        if not model_data.keep_training(iterator):
            break
        p, q = Sentence(batch, model_data,
                        args.data_type).generate(args.device)

        preds = model(p, q)

        optimizer.zero_grad()
        batch_loss = criterion(preds, batch.label)
        train_loss += batch_loss.data.item()
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_interval == 0:
            valid_loss, valid_acc = test(model, args, model_data, mode='valid')
            test_loss, test_acc = test(model, args, model_data)
            c = (i + 1) // args.print_interval

            writer.add_scalar('loss/train', train_loss, c)
            writer.add_scalar('loss/valid', valid_loss, c)
            writer.add_scalar('acc/valid', valid_acc, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/test', test_acc, c)

            print(
                f'train_loss: {train_loss:.3f}\n',
                f'valid_loss: {valid_loss:.3f}\n',
                f'test_loss: {test_loss:.3f}\n',
                f'valid_acc: {valid_acc:.3f}\n',
                f'test_acc: {test_acc:.3f}',
                sep='')

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            train_loss = 0
            model.train()

    print(
        f'max_valid_acc: {max_valid_acc:.3f}\n',
        f'max_test_acc: {max_test_acc:.3f}',
        sep='')
    writer.close()

    return best_model


if __name__ == '__main__':
    plac.call(main)

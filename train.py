"""A script to train and test a PyTorch implementation of the BiMPM model."""

import os
import copy
from time import gmtime, strftime
import plac

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from test import test
from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence, Args


def main(batch_size: ('[64]', 'positional', None, int) = 64,
         char_input_size: ('[20]', 'positional', None, int) = 20,
         char_hidden_size: ('[50]', 'positional', None, int) = 50,
         data_type: ("{SNLI, [Quora]}") = 'quora',
         dropout: ('[0.1]', 'positional', None, float) = 0.1,
         epoch: ('[10]', 'positional', None, int) = 10,
         hidden_size: ('[100]', 'positional', None, int) = 100,
         lr: ('[0.001]', 'positional', None, float) = 0.001,
         num_perspectives: ('[20]', 'positional', None, int) = 20,
         print_interval: ('[500]', 'positional', None, int) = 500,
         word_dim: ('[300]', 'positional', None, int) = 300):
    """Train and store the best BiMPM model in a cycle.

    Keyword arguments:
    batch_size -- number of examples in one iteration (default 64),
    char_input_size -- size of character embedding (default 20),
    char_hidden_size -- size of hidden layer in char lstm (default 50),
    data_type -- either SNLI or Quora (default 'quora'),
    dropout -- applied to each layer (default 0.1),
    epoch -- number of passes through full dataset (default 10),
    hidden_size -- size of hidden layer for all BiLSTM layers (default 100),
    lr -- learning rate (default 0.001),
    num_perspectives -- number of perspectives in matching layer (default 20),
    print_interval -- how often to write to tensorboard (default 500),
    word_dim -- size of word embeddings (default 300):
    """
    # Store local namespace dict in Args() object
    args = Args(locals())

    args.device = torch.device('cuda:0' if torch.cuda.
                               is_available() else 'cpu')

    if args.data_type.lower() == 'snli':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type.lower() == 'quora':
        print("Loading Quora data...")
        # model_data = Quora(args, toy=True) # Use for experimentation 
        model_data = Quora(args)
    else:
        raise RuntimeError(
            'Data source other than SNLI or Quora was provided.')

    # Create a few more parameters based on chosen dataset
    args.char_vocab_size = len(model_data.char_vocab)
    args.word_vocab_size = len(model_data.TEXT.vocab)
    args.class_size = len(model_data.LABEL.vocab)
    args.max_word_len = model_data.max_word_len
    args.model_time = strftime('%H:%M:%S', gmtime())

    print("Starting training...")
    best_model = train(args, model_data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(),
               f'saved_models/bimpm_{args.data_type}_{args.model_time}.pt')

    print("Finished training...")


def train(args, model_data):
    """Train BiMPM model on SNLI or Quora data.

    Keyword arguments:
    args -- Args() object with all arguments for BiMPM model
    model_data -- data loading object which returns word vectors and sentences
    """
    model = BiMPM(args, model_data)
    model.to(args.device)

    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize tensorboardx logging 
    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    train_loss, max_valid_acc, max_test_acc = 0, 0, 0

    iterator = model_data.train_iter
    for i, batch in enumerate(iterator):
        # Train for args.epoch number of epochs
        if not model_data.keep_training(iterator):
            break
        # Sentence() object contains chars and word batches
        p, q = Sentence(batch, model_data,
                        args.data_type).generate(args.device)

        preds = model(p, q)

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        batch_loss = criterion(preds, batch.label)
        train_loss += batch_loss.data.item()
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_interval == 0:
            valid_loss, valid_acc = test(model, args, model_data, mode='valid')
            test_loss, test_acc = test(model, args, model_data, mode='test')
            c = (i + 1) // args.print_interval  # Calculate step

            # Update tensorboardx logs
            writer.add_scalar('loss/train', train_loss, c)
            writer.add_scalar('loss/valid', valid_loss, c)
            writer.add_scalar('acc/valid', valid_acc, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/test', test_acc, c)

            print(
                f'\ntrain_loss:  {train_loss:.3f}\n',
                f'valid_loss:  {valid_loss:.3f}\n',
                f'test_loss:   {test_loss:.3f}\n',
                f'valid_acc:   {valid_acc:.3f}\n',
                f'test_acc:    {test_acc:.3f}\n',
                sep='')

            # Track best model and metrics so far
            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            train_loss = 0
            model.train()

    print(
        f'\nmax_valid_acc:  {max_valid_acc:.3f}\n',
        f'max_test_acc:   {max_test_acc:.3f}\n',
        sep='')
    writer.close()

    return best_model


if __name__ == '__main__':
    plac.call(main)  # Only executed when script is run directly

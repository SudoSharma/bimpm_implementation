"""Trains and tests a PyTorch implementation of the BiMPM model."""

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


def main(batch_size: (None, 'option', None, int) = 64,
         char_input_size: (None, 'option', None, int) = 20,
         char_hidden_size: (None, 'option', None, int) = 50,
         data_type: ("use quora or snli", 'option', None, str,
                     ['quora', 'snli']) = 'quora',
         dropout: (None, 'option', None, float) = 0.1,
         epoch: (None, 'option', None, int) = 10,
         hidden_size: (None, 'option', None, int) = 100,
         lr: (None, 'option', None, float) = 0.001,
         num_perspectives: (None, 'option', None, int) = 20,
         print_interval: (None, 'option', None, int) = 500,
         word_dim: (None, 'option', None, int) = 300,
         shutdown: ("shutdown system after training", 'option', None,
                    bool) = False):
    """Train and store the best BiMPM model in a cycle.

    Parameters
    ----------
    batch_size : int, optional
        Number of examples in one iteration (default is 64).
    char_input_size : int, optional
        Size of character embedding (default is 20).
    char_hidden_size : int, optional
        Size of hidden layer in char lstm (default is 50).
    data_type : {'Quora', 'SNLI'}, optional
        Choose either SNLI or Quora (default is 'quora').
    dropout : int, optional
        Applied to each layer (default is 0.1).
    epoch : int, optional
        Number of passes through full dataset (default is 10).
    hidden_size : int, optional
        Size of hidden layer for all BiLSTM layers (default is 100).
    lr : int, optional
        Learning rate (default is 0.001).
    num_perspectives : int, optional
        Number of perspectives in matching layer (default is 20).
    print_interval : int, optional
        How often to write to tensorboard (default is 500).
    word_dim : int, optional
        Size of word embeddings (default is 300).
    shutdown: bool, optional
        Whether or not to shutdown system after training (default is False).

    Raises
    ------
    RuntimeError
        If any data source other than SNLI or Quora is requested.

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
        # model_data = Quora(args, toy=True)  # Use for experimentation
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

    if args.shutdown:
        print("Shutting system down...")
        os.system("sudo shutdown now -h")


def train(args, model_data):
    """Train the BiMPM model on SNLI or Quora data.

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model.
    model_data : {Quora, SNLI}
        A data loading object which returns word vectors and sentences.

    Returns
    -------
    best_model : BiMPM
        The BiMPM model with the highest accuracy on the test set.

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

"""Trains and evaluates a PyTorch implementation of the BiMPM model."""

import os
import copy
from time import gmtime
import calendar
import plac

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from evaluate import evaluate
from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence, Args


def main(shutdown: ("shutdown system after training", 'flag', 's'),
         travis: ("use small testing dataset", 'flag', 't'),
         experiment: ("name of experiment", 'option', 'e', str) = '0.0',
         grad_clip: (None, 'option', None, int) = 100,
         batch_size: (None, 'option', None, int) = 64,
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
         word_dim: (None, 'option', None, int) = 300):
    """Train and store the best BiMPM model in a cycle.

    Parameters
    ----------
    shutdown : bool, flag
        Shutdown system after training (default is False).
    travis : bool, flag
        Run tests on small dataset (default is False).
    experiment : str, optional
        Name of the current experiment (default is '0.0').
    grad_clip : int, optional
        Amount by which to clip the gradient (default is 100).
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

    Raises
    ------
    RuntimeError
        If any data source other than SNLI or Quora is requested.

    """
    # Store local namespace dict in Args() object
    args = Args(locals())

    args.device = torch.device('cuda:0' if torch.cuda.
                               is_available() else 'cpu')

    args.app = False  # Disable app mode for training

    # Handle travis mode
    if args.travis and args.data_type.lower() == 'snli':
        raise RuntimeError("Invalid dataset size specified for SNLI data.")

    if args.travis:
        print('Travis mode detected. Adjusting parameters...')
        args.epoch = 2
        args.batch_size = 2
        args.print_interval = 1

    # Load data from sources
    if args.data_type.lower() == 'snli':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type.lower() == 'quora':
        print("Loading Quora data...")
        model_data = Quora(args)
    else:
        raise RuntimeError(
            'Data source other than SNLI or Quora was provided.')

    # Create a few more parameters based on chosen dataset
    args.char_vocab_size = len(model_data.char_vocab)
    args.word_vocab_size = len(model_data.TEXT.vocab)
    args.class_size = len(model_data.LABEL.vocab)
    args.max_word_len = model_data.max_word_len
    args.model_time = str(calendar.timegm(gmtime()))

    # Store hyperparameters for reproduceability
    if not os.path.exists('research/configs'):
        os.makedirs('research/configs')
    if not args.travis:
        args.store_params()

    print("Starting training...")
    best_model = train(args, model_data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    if not args.travis:
        torch.save(
            best_model.state_dict(),
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
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not args.travis:
        writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    train_loss, max_valid_acc, max_eval_acc = 0, 0, 0

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
        for p in parameters:
            # Clip gradients
            p.grad.data = p.grad.data.clamp(-args.grad_clip, args.grad_clip)
        optimizer.step()

        if (i + 1) % args.print_interval == 0:
            valid_loss, valid_acc = evaluate(
                model, args, model_data, mode='valid')
            eval_loss, eval_acc = evaluate(
                model, args, model_data, mode='eval')
            c = (i + 1) // args.print_interval  # Calculate step

            # Update tensorboardx logs
            if not args.travis:
                writer.add_scalar('loss/train', train_loss, c)
                writer.add_scalar('loss/valid', valid_loss, c)
                writer.add_scalar('acc/valid', valid_acc, c)
                writer.add_scalar('loss/eval', eval_loss, c)
                writer.add_scalar('acc/eval', eval_acc, c)

            print(
                f'\ntrain_loss:  {train_loss:.3f}\n',
                f'valid_loss:  {valid_loss:.3f}\n',
                f'eval_loss:   {eval_loss:.3f}\n',
                f'valid_acc:   {valid_acc:.3f}\n',
                f'eval_acc:    {eval_acc:.3f}\n',
                sep='')

            # Track best model and metrics so far
            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                max_eval_acc = eval_acc
                best_model = copy.deepcopy(model)

            train_loss = 0
            model.train()

    print(
        f'\nmax_valid_acc:  {max_valid_acc:.3f}\n',
        f'max_eval_acc:   {max_eval_acc:.3f}\n',
        sep='')
    if not args.travis:
        writer.close()

    return best_model


if __name__ == '__main__':
    plac.call(main)  # Only executed when script is run directly

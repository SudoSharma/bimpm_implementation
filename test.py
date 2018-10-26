"""Tests a model trained on a PyTorch reimplementation of BiMPM"""

import plac

import torch
from torch import nn

from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence, Args


def main(model_path,
         batch_size: (None, 'option', None, int) = 64,
         char_input_size: (None, 'option', None, int) = 20,
         char_hidden_size: (None, 'option', None, int) = 50,
         data_type: ("quora or snli", 'option', None, str,
                     ['quora', 'snli']) = 'quora',
         dropout: (None, 'option', None, float) = 0.1,
         epoch: (None, 'option', None, int) = 10,
         hidden_size: (None, 'option', None, int) = 100,
         lr: (None, 'option', None, float) = 0.001,
         num_perspectives: (None, 'option', None, int) = 20,
         print_interval: (None, 'option', None, int) = 500,
         word_dim: (None, 'option', None, int) = 300):
    """Print the best BiMPM model accuracy for the test set in a cycle.

    Parameters
    ----------
    model_path : str
        A path to the location of the BiMPM trained model.
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

    if args.data_type == 'SNLI':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type == 'Quora':
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

    print("Loading model...")
    model = load_model(args, model_data)

    _, test_acc = test(model, args, model_data, mode='test')

    print(f'\ntest_acc:  {test_acc:.3f}\n')


def test(model, args, model_data, mode='test'):
    """Test the BiMPM model on SNLI or Quora validation or test data.

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model.
    model_data : {Quora, SNLI}
        A data loading object which returns word vectors and sentences.
    mode : int, optional
        Indicates whether or not to use valid or test data (default is 'test').

    Returns
    -------
    loss : int
        The loss of the model provided.
    acc : int
        The accuracy of the model provided.

    """
    if mode == 'valid':
        iterator = model_data.valid_iter
    elif mode == 'test':
        iterator = model_data.test_iter

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        # Sentence() object contains chars and word batches
        p, q = Sentence(batch, model_data,
                        args.data_type).generate(args.device)

        preds = model(p, q)
        batch_loss = criterion(preds, batch.label)
        loss += batch_loss.data.item()

        # Retrieve index of class with highest score and calculate accuracy
        _, preds = preds.max(dim=1)
        acc += (preds == batch.label).sum().float()
        size += len(preds)

    acc /= size
    acc = acc.cpu().data.item()
    return loss, acc


def load_model(args, model_data):
    """Load the trained BiMPM model for testing

    Parameters
    ----------
    args : Args
        An object with all arguments for BiMPM model
    model_data : {Quora, SNLI}
        A data loading object which returns word vectors and sentences.

    Returns
    -------
    model : BiMPM
        A new model initialized with the weights from the provided trained
        model.
    """
    model = BiMPM(args, model_data)
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)

    return model


if __name__ == '__main__':
    plac.call(main)  # Only executed when script is run directly

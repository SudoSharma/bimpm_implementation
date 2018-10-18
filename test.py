from collections import namedtuple
import plac

import torch
from torch import nn

from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence


def main(model_path,
         batch_size=64,
         char_dim=20,
         char_hidden_size=50,
         data_type=(['SNLI', 'Quora']),
         dropout=0.1,
         epoch=10,
         gpu=True,
         hidden_size=100,
         lr=0.001,
         num_perspectives=20,
         word_dim=300):
    args = locals()

    if args.data_type == 'SNLI':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type == 'Quora':
        print("Loading Quoradata...")
        model_data = Quora(args)
    else:
        raise RuntimeError(
            'Data source other than SNLI or Quora was provided.')

    args['char_vocab_size'] = len(model_data.char_vocab)
    args['word_vocab_size'] = len(model_data.TEXT.vocab)
    args['class_size'] = len(model_data.LABEL.vocab)
    args['max_word_len'] = model_data.max_word_len

    args = namedtuple('args', args.keys())(*args.values())

    print("Loading model...")
    model = load_model(args, model_data)

    _, test_acc = test(model, args, model_data)

    print(f'test_acc: {test_acc:.3f}')


def test(model, args, model_data, mode='test'):
    if mode == 'valid':
        iterator = iter(model_data.valid_iter)
    elif mode == 'test':
        iterator = iter(model_data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        p, q = Sentence(batch, model_data, args.data_type).generate(args.gpu)

        preds = model(p, q)

        batch_loss = criterion(preds, batch.label)
        loss += batch_loss.data[0]

        _, preds = preds.max(dim=1)
        acc += (preds == batch.label).sum().float()
        size += len(preds)

    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc


def load_model(args, model_data):
    model = BiMPM(args, model_data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu:
        model.cuda(args.gpu)

    return model


if __name__ == '__main__':
    plac.call(main)

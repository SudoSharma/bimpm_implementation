import plac

import torch
from torch import nn

from model.bimpm import BiMPM
from model.utils import SNLI, Quora, Sentence, Args


def main(model_path,
         batch_size: ('[64]', 'positional', None, int)=64,
         char_input_size: ('[20]', 'positional', None, int)=20,
         char_hidden_size: ('[50]', 'positional', None, int)=50,
         data_type: ("{SNLI, [Quora]}")='quora',
         dropout: ('[0.1]', 'positional', None, float)=0.1,
         epoch: ('[10]', 'positional', None, int)=10,
         hidden_size: ('[100]', 'positional', None, int)=100,
         lr: ('[0.001]', 'positional', None, float)=0.001,
         num_perspectives: ('[20]', 'positional', None, int)=20,
         word_dim: ('[300]', 'positional', None, int)=300):
    args = Args(locals())

    args.device = torch.device('cuds:0' if torch.cuda.
                               is_available() else 'cpu')

    if args.data_type == 'SNLI':
        print("Loading SNLI data...")
        model_data = SNLI(args)
    elif args.data_type == 'Quora':
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
        p, q = Sentence(batch, model_data,
                        args.data_type).generate(args.device)

        preds = model(p, q)

        batch_loss = criterion(preds, batch.label)
        loss += batch_loss.data.item()

        _, preds = preds.max(dim=1)
        acc += (preds == batch.label).sum().float()
        size += len(preds)

    acc /= size
    acc = acc.cpu().data.item()
    return loss, acc


def load_model(args, model_data):
    model = BiMPM(args, model_data)
    model.load_state_dict(torch.load(args.model_path))

    model.to(args.device)

    return model


if __name__ == '__main__':
    plac.call(main)

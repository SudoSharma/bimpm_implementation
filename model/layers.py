import torch
import torch.nn as nn
import torch.nn.Functional as F
import plac


def main():
    pass


class CharacterRepresentationEncoder(nn.Module):
    def __init__(self, args):
        super(CharacterRepresentationEncoder, self).__init__()

        self.args = args

        self.char_encoder = nn.Embedding(
                args.char_vocab_size, args.char_dim, padding_idx=0)
        self.lstm = nn.LSTM(
                input_size=args.char_input_size,
                hidden_size=args.char_hidden_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True)

    def forward(self, chars):
        batch_size, seq_len, max_word_len = chars.size()
        chars = chars.view(batch_size*seq_len, max_word_len)
        _, (chars, _) = self.lstm(self.char_encoder(chars))

        return chars.view(-1, seq_len, self.args.char_hidden_size)


class WordRepresentationLayer(nn.Module):
    def __init__(self, args, data):
        super(WordRepresentationLayer, self).__init__()

        self.args = args

        self.word_encoder = nn.Embedding(args.word_vocab_size, args.word_dim)
        self.word_encoder.weight.data.copy_(data.TEXT.vocab.vectors)
        self.word_encoder.weight.requires_grad = False

        self.char_encoder = CharacterRepresentationEncoder(args)

    def droput(self, V):
        return F.droput(V, p=self.args.dropout, training=self.training)

    def forward(self, sentence):
        words = sentence.words
        chars = self.char_encoder(sentence.chars)
        sentence = torch.cat([words, chars], dim=-1)

        return self.dropout(sentence)

class ContextRepresentationLayer(nn.Module):
    def __init__(self, args):
        super(ContextRepresentationLayer, self).__init__()

        self.input_size = args.word_dim + args.char_hidden_size

        self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=args.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True)

    def droput(self, V):
        return F.droput(V, p=self.args.dropout, training=self.training)

    def forward(self, sentence):
        sentence, _ = self.lstm(sentence)

        return self.dropout(sentence)


class MatchingLayer(nn.Module):
    def __init__(self):
        super(MatchingLayer, self).__init__()
        pass

    def forward(self):
        pass


class AggregationLayer(nn.Module):
    def __init__(self):
        super(AggregationLayer, self).__init__()
        pass

    def forward(self):
        pass


class PredictionLayer(nn.Module):
    def __init__(self):
        super(PredictionLayer, self).__init__()
        pass

    def forward(self):
        pass


if __name__() == "__main__":
    plac.call(main)

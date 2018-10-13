import torch
import torch.nn as nn
import torch.nn.Functional as F
import plac


def main():
    pass


class CharacterRepresentationEncoder(nn.Module):
    def __init__(self, args):
        super(CharacterRepresentationEncoder, self).__init__()

        self.char_hidden_size = args.char_hidden_size

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
        chars = self.lstm(self.char_encoder(chars))[-1][0]

        return chars.view(-1, seq_len, self.char_hidden_size)


class WordRepresentationLayer(nn.Module):
    def __init__(self, args, data):
        super(WordRepresentationLayer, self).__init__()

        self.drop = args.dropout

        self.word_encoder = nn.Embedding(args.word_vocab_size, args.word_dim)
        self.word_encoder.weight.data.copy_(data.TEXT.vocab.vectors)
        self.word_encoder.weight.requires_grad = False

        self.char_encoder = CharacterRepresentationEncoder(args)

    def dropout(self, V):
        return F.droput(V, p=self.drop, training=self.training)

    def forward(self, sentence):
        words = sentence.words
        chars = self.char_encoder(sentence.chars)
        sentence = torch.cat([words, chars], dim=-1)

        return self.dropout(sentence)


class ContextRepresentationLayer(nn.Module):
    def __init__(self, args):
        super(ContextRepresentationLayer, self).__init__()

        self.drop = args.dropout
        self.input_size = args.word_dim + args.char_hidden_size

        self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=args.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True)

    def dropout(self, V):
        return F.dropout(V, p=self.drop, training=self.training)

    def forward(self, sentence):
        sentence = self.lstm(sentence)[0]

        return self.dropout(sentence)


class MatchingLayer(nn.Module):
    def __init__(self, args):
        super(MatchingLayer, self).__init__()

        self.hidden_size = args.hidden_size
        self.l = args.num_perspectives
        self.W = nn.ParameterList([nn.Parameter(
            torch.rand(self.l, self.hidden_size)) for _ in range(8)])

    def full_match(self, p, q, w):
        pass

    def maxpool_match(self, p, q, w):
        pass

    def attentive_match(self, p, q, w):
        pass

    def max_attentive_match(self, p, q, w):
        pass

    def cat(self, *args):
        return torch.cat(list(args), dim=2)

    def match_operation(self, p, q, W):
        full_p2q_fw = self.full_match(p, q[-1], W[0])
        full_p2q_bw = self.full_match(p, q[0], W[1])
        full_q2p_fw = self.full_match(q, p[-1], W[0])
        full_q2p_bw = self.full_match(q, p[0], W[1])

        pool_p2q_fw = self.maxpool_match(p, q, W[2])
        pool_p2q_bw = self.maxpool_match(p, q, W[3])
        pool_q2p_fw = self.maxpool_match(q, p, W[2])
        pool_q2p_bw = self.maxpool_match(q, p, W[3])

        att_p2mean_fw = self.attentive_match(p, q, W[5])
        att_p2mean_bw = self.attentive_match(p, q, W[6])
        att_q2mean_fw = self.attentive_match(p, q, W[5])
        att_p2mean_bw = self.attentive_match(p, q, W[6])

        max_att_p2max_fw = self.max_attentive_match(p, q, W[5])
        max_att_p2max_bw = self.max_attentive_match(p, q, W[6])
        max_att_q2max_fw = self.max_attentive_match(p, q, W[5])
        max_att_p2max_bw = self.max_attentive_match(p, q, W[6])

        p_vec = self.cat(
                full_p2q_fw, pool_p2q_fw, att_p2mean_fw, max_att_p2max_fw,
                full_p2q_bw, pool_p2q_bw, att_p2mean_bw, max_att_p2max_bw)

        q_vec = self.cat(
                full_q2p_fw, pool_q2p_fw, att_q2mean_fw, max_att_q2max_fw,
                full_q2p_bw, pool_q2p_bw, att_q2mean_bw, max_att_q2max_bw)

        return (self.dropout(p_vec), self.dropout(q_vec))

    def forward(self, p, q):
        return  match_operation(p, q, self.W)


class AggregationLayer(nn.Module):
    def __init__(self, args):
        super(AggregationLayer, self).__init__()

        self.hidden_size = args.hidden_size
        self.drop = args.dropout
        self.lstm = self.LSTM(
                input_size=args.num_perpectives*8,
                hidden_size=args.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True)

    def dropout(self, V):
        return F.dropout(V, p=self.drop, training=self.training)

    def forward(self, p, q):
        p = self.lstm(p)[-1][0]
        q = self.lstm(q)[-1][0]

        x = torch.cat(
                [p.permute(1, 0, 2).view(-1, self.hidden_size*2),
                 q.permute(1, 0, 2).view(-1, self.hidden_size*2)], dim=1)

        return self.dropout(x)


class PredictionLayer(nn.Module):
    def __init__(self, args):
        super(PredictionLayer, self).__init__()

        self.drop = args.dropout
        self.hidden_layer = nn.Linear(args.hidden_size*4, args.hidden_size*2)
        self.output_layer = nn.Linear(args.hidden_size*2, args.num_classes)

    def dropout(self, V):
        return F.dropout(V, p=self.drop, training=self.training)

    def forward(self, match_vec):
        x = F.relu(self.hidden_layer(match_vec))

        return F.softmax(self.output_layer(self.dropout(x)))


if __name__() == "__main__":
    plac.call(main)

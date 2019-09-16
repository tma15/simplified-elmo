import torch
import torch.nn.functional as F

from .elmo import ElmoBiLm


class LanguageModel(torch.nn.Module):
    def __init__(self, vocab, hidden_size=30):
        super(LanguageModel, self).__init__()

        self.hidden_size = hidden_size
        self.elmo = ElmoBiLm(hidden_size=hidden_size)
        self.out = torch.nn.Linear(hidden_size, len(vocab))

    def get_activations(self, inputs):
        o = self.elmo(inputs)
        return o

    def forward(self, inputs):
        hidden = self.get_activations(inputs)

        os = torch.chunk(hidden['activations'][-1], 2, dim=2)
        for o in os:
            p = self.out(o)

    def forward_loss(self, inputs, targets):
        # inputs: (batch, sentence_length, token_length)
        # targets: (batch, sentence_length)

        batch, sentence_length, _ = inputs.size()

        hidden = self.get_activations(inputs)

        # (batch, sentence_length, 2 * hidden_size)
        hidden = hidden['activations'][-1]

        # (batch, sentence_length, 2, hidden_size)
        hidden = hidden.contiguous().view(batch, sentence_length, 2, self.hidden_size)

        # (batch, 2, sentence_length, hidden_size)
        hidden = hidden.transpose(1, 2)

        # (2 * batch, sentence_length, hidden_size)
        hidden = hidden.contiguous().view(2 * batch, sentence_length, self.hidden_size)

        # (2 * batch, sentence_length)
        targets = targets.repeat(2, 1)

        loss = 0
        for t in range(sentence_length):
            h_t = self.out(hidden[:, t, :])
            loss += F.nll_loss(F.log_softmax(h_t, dim=1), targets[:, t])

        return loss

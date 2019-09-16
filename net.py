import torch
import torch.nn.functional as F

from elmo.elmo import Elmo


class Classifier(torch.nn.Module):
    def __init__(self, hidden_size, class_size):
        super(Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.class_size = class_size

        self.elmo = Elmo(hidden_size=hidden_size)

        self.embed = torch.nn.Embedding(262, 2 * self.hidden_size)

        self.lstm = torch.nn.LSTM(2 * self.hidden_size,
                                  self.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)

        self.out = torch.nn.Linear(2 * self.hidden_size, self.class_size)

    def load_elmo_weight(self, elmo_weight_file):
        self.elmo.elmo_bilm.load_state_dict(elmo_weight_file)

    def pre_softmax(self, inputs):
        # (batch, sentence_length, 2 * hidden_size)
#         h = self.elmo(inputs)
#         out, (h, c) = self.lstm(h)

        # (batch, sentence_length * token_length, 2 * hidden_size))
        w = self.embed(inputs.view(inputs.size(0), -1))
        # (2, batch, hidden_size))
        _, (h, c) = self.lstm(w)

        # (batch, 2 * hidden_size))
        h = h.transpose(0, 1).contiguous().view(h.size(1), -1)

        y = self.out(h)
        return y

    def forward(self, inputs):
        y = self.pre_softmax(inputs)
        _, argmax = torch.max(y, dim=1)
        return argmax

    def forward_loss(self, inputs, targets):
        y = self.pre_softmax(inputs)
        loss = F.nll_loss(F.log_softmax(y, dim=1), targets)
        return loss

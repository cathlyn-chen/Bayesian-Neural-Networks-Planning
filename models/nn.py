import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, hp):
        super(NN, self).__init__()
        self.input = nn.Linear(hp.n_input, hp.hidden_units)
        self.output = nn.Linear(hp.hidden_units, hp.n_output)

        # nn.init.xavier_uniform_(self.input.weight)
        # nn.init.zeros_(self.input.bias)
        # nn.init.xavier_uniform_(self.output.weight)
        # nn.init.zeros_(self.output.bias)

    def forward(self, x):
        out = F.sigmoid(self.input(x))
        out = self.output(out)
        return out
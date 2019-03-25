import torch
from torch import nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.lin = nn.Linear(self.hidden_size * 2, hidden_size * 2)
        self.weight_vec = nn.Parameter(torch.FloatTensor(1, hidden_size * 2))

    def forward(self, outputs):
        seq_len = len(outputs)

        attn_energies = torch.zeros(seq_len).cuda()  # B x 1 x S

        for i in range(seq_len):
            attn_energies[i] = self.score(outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, output):
        energy = self.lin(output)
        energy = torch.dot(self.weight_vec.view(-1), energy.view(-1))
        return energy

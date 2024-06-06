import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

class Attention(nn.Module):
    def __init__(self, nb_head, size_per_head):
        super(Attention, self).__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        
        # Initialize weights
        self.WQ = nn.Linear(size_per_head, self.output_dim, bias=False)
        self.WK = nn.Linear(size_per_head, self.output_dim, bias=False)
        self.WV = nn.Linear(size_per_head, self.output_dim, bias=False)

    def mask(self, inputs, seq_len, mode='mul'):
        if seq_len is None:
            return inputs
        else:
            mask = torch.ones_like(inputs)
            for i in range(mask.shape[0]):
                mask[i, seq_len[i]:] = 0
            if mode == 'mul':
                return inputs * mask
            elif mode == 'add':
                return inputs - (1 - mask) * 1e12

    def forward(self, Q_seq, K_seq, V_seq, Q_len=None, V_len=None):
        Q_seq = self.WQ(Q_seq)
        Q_seq = Q_seq.view(-1, Q_seq.shape[1], self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        
        K_seq = self.WK(K_seq)
        K_seq = K_seq.view(-1, K_seq.shape[1], self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        
        V_seq = self.WV(V_seq)
        V_seq = V_seq.view(-1, V_seq.shape[1], self.nb_head, self.size_per_head).permute(0, 2, 1, 3)
        
        A = torch.matmul(Q_seq, K_seq.transpose(-1, -2)) / (self.size_per_head ** 0.5)
        A = self.mask(A, V_len, 'add')
        A = F.softmax(A, dim=-1)
        
        O_seq = torch.matmul(A, V_seq)
        O_seq = O_seq.permute(0, 2, 1, 3).contiguous().view(-1, O_seq.shape[2], self.output_dim)
        O_seq = self.mask(O_seq, Q_len, 'mul')
        
        return O_seq

class AttentivePooling(nn.Module):
    def __init__(self, dim1, dim2):
        super(AttentivePooling, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.dense_tanh = nn.Linear(dim2, 200)
        self.dense_flat = nn.Linear(200, 1)
        
    def forward(self, x):
        x = self.dropout(x)
        att = torch.tanh(self.dense_tanh(x))
        att = self.dense_flat(att).squeeze(-1)
        att = F.softmax(att, dim=-1)
        att = att.unsqueeze(-1)
        output = torch.sum(x * att, dim=1)
        return output

class AttentivePoolingQKY(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(AttentivePoolingQKY, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.dense_tanh = nn.Linear(dim2, 200)
        self.dense_flat = nn.Linear(200, 1)

    def forward(self, x, y):
        x = self.dropout(x)
        att = torch.tanh(self.dense_tanh(x))
        att = self.dense_flat(att).squeeze(-1)
        att = F.softmax(att, dim=-1)
        att = att.unsqueeze(-1)
        output = torch.sum(y * att, dim=1)
        return output

class AttentivePoolingBias(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(AttentivePoolingBias, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.dense_tanh = nn.Linear(dim2, 200)
        self.dense_flat = nn.Linear(200, 1)

    def forward(self, bias_input, value_input):
        bias_input = self.dropout(bias_input)
        att = torch.tanh(self.dense_tanh(bias_input))
        att = self.dense_flat(att).squeeze(-1)
        att = F.softmax(att, dim=-1)
        att = att.unsqueeze(-1)
        output = torch.sum(value_input * att, dim=1)
        return output

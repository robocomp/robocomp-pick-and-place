import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_imputs, n_outputs, device):
        super(DQN, self).__init__()
        self.device = device
        self.l1 = nn.Linear(n_imputs, 36)
        self.l2 = nn.Linear(36, 25)
        self.l3 = nn.Linear(25, 16)
        self.l4 = nn.Linear(16, n_outputs)
        self.s = nn.Softmax()

    def forward(self, x):
        # print("ENTRADA", x.shape, x)
        x = torch.Tensor(x).to(self.device)
        # print("TENSOR", x.shape)
        x = F.relu(self.l1(x))
        # print("L1", x.shape)
        x = F.relu(self.l2(x))
        # print("L2", x.shape)
        x = F.relu(self.l3(x))
        # print("L3", x.shape)
        x = F.relu(self.l4(x))
        # print("L4", x.shape)
        x = self.s(x)
        # print("S", x.shape)
        return x



from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
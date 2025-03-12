import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc1_value = nn.Linear(hidden_dim, 256)
            # self.fc2_value = nn.Linear(256, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc1_advantages = nn.Linear(hidden_dim, 256)
            # self.fc2_advantages = nn.Linear(256, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            # self.output = nn.Linear(hidden_dim, action_dim)
            self.input = nn.Linear(hidden_dim, 256)
            self.output = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc1_value(x))
            # v = F.relu(self.fc2_value(v))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc1_advantages(x))
            # a = F.relu(self.fc2_advantages(a))
            A = self.advantages(a)

            # Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            # Q = self.output(x)
            Q = F.relu(self.output(F.relu(self.input(x))))

        return Q


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)


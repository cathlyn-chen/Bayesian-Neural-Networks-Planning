import torch
from torch import nn

from ..eval.eval_reg import *


class RNNCell:
    def __init__(self, problem):
        """
        :param problem:     (string) Name of current problem
        """
        self.problem = problem

    def forward(self, state, action):
        """
        :param state:       (pytorch.tensor, shape[batch_size, 2])  An (x,y) coordinate for current state
        :param action:      (pytorch.tensor, shape[batch_size, 2])  A (dx, dy) action to move to next state
        :return:            (pytorch.tensor, shape[batch_size])     Reward for each batch
                            (pytorch.tensor, shape[2]) An (x,y)     An (x,y) coordinate for next state
        """

        next_state = self.problem.TransitionModel(state, action)
        reward = self.problem.RewardFunction(next_state, action)

        return reward, next_state


class RNNClass(nn.Module):
    def __init__(self, problem):
        super(RNNClass, self).__init__()

        self.problem = problem

        self.actions = torch.nn.Parameter(
            torch.zeros(self.problem.actions_shape,
                        device=torch.device('cuda:0'),
                        requires_grad=True))
        torch.nn.init.xavier_normal_(self.actions,
                                     gain=self.problem.xavier_gain)

        self.problem.UpdateActions(self)

        self.rnn_cell_list = []
        for i in range(self.problem.horizon):
            self.rnn_cell_list.append(RNNCell(self.problem))

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.problem.learning_rate_val)

    def forward(self):
        """
        :return:    reward:                 (pytorch.tensor, shape[time_horizons, batch_size])      Reward corresponding to each batch and time
        """

        self.optimizer.zero_grad()

        reward = torch.zeros((self.problem.horizon, self.problem.batch_size),
                             device=torch.device('cuda:0'))
        next_states = torch.zeros(self.problem.next_states_size,
                                  device=torch.device('cuda:0'))
        actions = torch.repeat_interleave(torch.unsqueeze(self.actions, 1),
                                          self.problem.batch_size,
                                          dim=1)

        for i in range(self.problem.horizon):
            if i == 0:
                reward[i, :], next_states[i] = self.rnn_cell_list[i].forward(
                    self.problem.init_state, actions[i])
            else:
                reward[i, :], next_states[i] = self.rnn_cell_list[i].forward(
                    next_states[i - 1], actions[i])

        self.problem.FinishForward(next_states)

        return reward

    def backward(self, rewards, epoch):
        Loss = self.problem.CalculateLoss(rewards, epoch)

        # Compute auto grad on computation graph to get the gradient for the actions
        Loss.backward()

        # Take optimizer step
        self.optimizer.step()

        self.problem.UpdateActions(self)

        return Loss


class NAV_PROBLEM():
    def __init__(self, loss_function, net_x, net_y, hp):
        self.loss_function = loss_function
        self.hp = hp
        self.net_x = net_x
        self.net_y = net_y

    def TransitionModel(self, state, action):
        x, y = state[0], state[1]
        a1, a2 = action[0], action[1]

        _, next_x, uncertain_x = eval_reg(self.net_x, x, self.hp.pred_samples)

        _, next_y, uncertain_y = eval_reg(self.net_y, y, self.hp.pred_samples)

        next_state = [next_x, next_y]
        uncertainty = uncertain_x + uncertain_y
        return next_state, uncertainty

    def RewardFunction(self):
        pass
import torch
from ..utils.plot import plot_states


def nav_data(hp):
    plot_states(hp)


class NAV:
    def __init__(self, hp):
        self.init_state = hp.init_state
        self.goal_state = hp.goal_state

        self.hp = hp

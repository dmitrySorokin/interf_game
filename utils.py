import numpy as np


def action2vec(action_id, n_actions):
    res = np.zeros(n_actions - 1)
    if action_id < n_actions - 1:
        res[action_id] = 1
    return res


def rescale_visib(visib):
    return -np.log(1.0 - visib + 1e-3) / 10.0 + visib

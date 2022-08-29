import torch
import torch.nn as nn
import cv2
import numpy as np


class DuelDQNModel(nn.Module):
    """A Dueling DQN net"""

    def __init__(self, input_shape, n_actions):
        super(DuelDQNModel, self).__init__()

        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))

    def forward(self, x):
        x = x.float() / 255.
        conv_out = self.conv(x).view(x.shape[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


class DQNAgent:
    """
    Simple DQNAgent which calculates Q values from list of observations
                          calculates actions given np.array of qvalues
    """

    def __init__(self, model_path, observation_shape, action_shape, step_fractions=(0.01, 0.05, 0.1)):
        self.step_fractions = step_fractions
        self.ndir = action_shape[0] * 2
        self.action_shape = action_shape[0]
        self.n_actions = self.ndir * len(self.step_fractions) + 1
        self.cont_action_scale = np.ones(self.action_shape)

        print('in_channels = {}, n_actions = {}'.format(observation_shape, self.n_actions))

        dqn_model = DuelDQNModel(observation_shape, self.n_actions)
        state = torch.load(model_path, map_location='cpu')
        dqn_model.load_state_dict(state)
        dqn_model.eval()

        self.dqn_model = dqn_model
        self.obs_shape = observation_shape

    def get_q_values(self, state):
        """
        Calculates q-values given list of obseravations
        """

        state = self._np2torch(state)
        q_values = self.dqn_model.forward(state)

        return q_values.detach().cpu().numpy()

    def _np2torch(self, state):
        return torch.from_numpy(state).unsqueeze(dim=0)

    def get_discrete_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)

        qvalues = self.get_q_values(state)
        action_id = qvalues.argmax(axis=-1)[0]
        return action_id

    def get_action(self, state):
        """
        Pick actions given array of qvalues
        Uses epsilon-greedy exploration strategy
        """

        action_id = self.get_discrete_action(state, 0)

        return self._to_cont_action(action_id)

    def _to_discrete_action(self, cont_action):
        cont_action /= self.cont_action_scale
        nonzero = np.nonzero(cont_action)[0]
        if len(nonzero) == 0:
            return self.n_actions - 1

        assert len(nonzero) == 1, 'THIS AGENT MAKE ONLY ONE ACTION AT TIME'
        nonzero_id = nonzero[0]

        action_id = nonzero_id * 2
        if cont_action[nonzero_id] > 0:
            action_id += 1

        step_fraction_id = np.where(self.step_fractions == np.abs(cont_action[nonzero_id]))[0][0]

        return action_id + step_fraction_id * self.ndir

    def _to_cont_action(self, action_id):
        actions = np.zeros(self.action_shape)

        if action_id < self.n_actions - 1:
            step_fraction = self.step_fractions[action_id // self.ndir]
            action_id = action_id % self.ndir
            actions[action_id // 2] = step_fraction * (
                -1 if action_id % 2 == 0 else 1
            )

        return actions * self.cont_action_scale

    def update_rhs(self, visib, action):
        pass

    def reset(self):
        pass

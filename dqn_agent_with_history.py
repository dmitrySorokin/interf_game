import torch
import torch.nn as nn
import numpy as np
from utils import action2vec, rescale_visib


class DuelDQNModelWithHistory(nn.Module):
    """A Dueling DQN net"""

    def __init__(self, input_shape, n_actions, history_length, history_in_size, history_out_size):
        super(DuelDQNModelWithHistory, self).__init__()

        self.n_actions = n_actions
        self.history_length = history_length

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.history = nn.Sequential(
            nn.LSTM(history_in_size, history_out_size)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size + history_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size + history_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))

    def forward(self, x, histo):
        x = x.float() / 255.
        conv_out = self.conv(x).view(x.shape[0], -1)

        histo = histo.view([histo.shape[0], -1, self.history_length]).permute([2, 0, 1])
        self.history[0].flatten_parameters()
        histo_out, hidden = self.history(histo)
        histo_out = histo_out[-1]
        out = torch.cat([conv_out, histo_out], dim=1)

        val = self.fc_val(out)
        adv = self.fc_adv(out)
        return val + adv - adv.mean()


class DQNAgentWithHistory:
    """
    Simple DQNAgent which calculates Q values from list of observations
                          calculates actions given np.array of qvalues
    """

    def __init__(
            self,
            model_path,
            observation_shape,
            step_fractions=(0.01, 0.05, 0.1),
            history_length=100
        ):
        self.step_fractions = step_fractions
        self.n_actions = 8 * len(self.step_fractions) + 1

        print('in_channels = {}, n_actions = {}'.format(observation_shape, self.n_actions))

        history_step_size = self.n_actions
        self.histo = np.zeros(history_step_size * history_length, dtype=np.float32)
        dqn_model = DuelDQNModelWithHistory(
            observation_shape, self.n_actions, history_length, history_step_size, history_step_size)

        state = torch.load(model_path, map_location='cpu')
        dqn_model.load_state_dict(state)
        dqn_model.eval()

        self.dqn_model = dqn_model
        self.obs_shape = observation_shape

        self.histo = np.zeros(history_length * history_step_size, dtype=np.float32)

    def get_q_values(self, state, histo):
        """
        Calculates q-values given list of obseravations
        """

        state = self._np2torch(state)
        histo = self._np2torch(histo)
        q_values = self.dqn_model.forward(state, histo)

        return q_values.detach().cpu().numpy()

    def _np2torch(self, array):
        return torch.from_numpy(array).unsqueeze(dim=0)

    def get_discrete_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)

        qvalues = self.get_q_values(state, self.histo)
        action_id = qvalues.argmax(axis=-1)[0]
        return action_id

    def get_action(self, state):
        """
        Pick actions given array of qvalues
        Uses epsilon-greedy exploration strategy
        """

        action_id = self.get_discrete_action(state, 0)
        return self._to_cont_action(action_id)

    def update_rhs(self, visib, action):
        rescaled_visib = rescale_visib(visib)
        discrete_action = self._to_discrete_action(action)
        action_vec = action2vec(discrete_action, self.n_actions)
        self.histo = np.append(self.histo[1 + len(action_vec):], [rescaled_visib, *action_vec]).astype(self.histo.dtype)

    def reset(self):
        self.histo = np.zeros_like(self.histo)

    def _to_discrete_action(self, cont_action):
        nonzero = np.nonzero(cont_action)[0]
        if len(nonzero) == 0:
            return self.n_actions - 1

        assert len(nonzero) == 1, 'THIS AGENT MAKE ONLY ONE ACTION AT TIME'
        nonzero_id = nonzero[0]

        action_id = nonzero_id * 2
        if cont_action[nonzero_id] > 0:
            action_id += 1

        step_fraction_id = np.where(self.step_fractions == np.abs(cont_action[nonzero_id]))[0][0]

        return action_id + step_fraction_id * 8

    def _to_cont_action(self, action_id):
        actions = [0] * 4

        if action_id < self.n_actions - 1:
            step_fraction = self.step_fractions[action_id // 8]
            action_id = action_id % 8
            actions[action_id // 2] = step_fraction * (
                -1 if action_id % 2 == 0 else 1
            )

        return actions

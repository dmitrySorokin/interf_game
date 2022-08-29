import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Beta


class MyActivationFunction(nn.Module):
    def __init__(self):
        super(MyActivationFunction, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class CCPPOActorCritic(nn.Module):
    # Changed Body

    def __init__(self, state_dim, action_dim, n_latent_var, activation=nn.Tanh(), action_h_dim=512):
        super().__init__()

        # Changed Body
        self.body = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.body_out_shape = self._get_layer_out(self.body, state_dim)

        self.nonlinearity = nn.Sequential(
            nn.Linear(self.body_out_shape, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh()
        )

        # Alfa head
        self.alfa_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            MyActivationFunction()
        )

        # Beta layer
        self.beta_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            MyActivationFunction()
        )

        # Critic head
        self.value_layer = nn.Sequential(
            nn.Linear(self.body_out_shape, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1)
        )


    def _get_layer_out(self, layer, shape):
        o = layer(torch.zeros(1, *shape))
        return int(np.prod(o.shape))

    def forward(self, state):
        state = state / 255
        state = torch.from_numpy(state).float()

        state = self.body(state)
        state = self.nonlinearity(state)

        alfa = self.alfa_layer(state)
        beta = self.beta_layer(state)
        dist = Beta(alfa, beta)

        return dist


class CCPPOAgent:
    def __init__(self, model_path, observation_shape, action_dim, n_latent_var):
        self.action_dim = action_dim

        print('in_channels = {}, action_dim = {}'.format(observation_shape, self.action_dim))

        ccppo_model = CCPPOActorCritic(observation_shape, self.action_dim, n_latent_var)
        state = torch.load(model_path, map_location='cpu')
        ccppo_model.load_state_dict(state)
        ccppo_model.eval()

        self.ccppo_model = ccppo_model
        self.obs_shape = observation_shape

    def get_action(self, state):

        dist = self.ccppo_model(np.array([state]))
        action = dist.mean.detach().cpu().numpy()[0] * 2 - 1
        res_act = np.array([0 if abs(a) < 0.5 else 10 ** (a-3) if a > 0 else -(10 ** (-a - 3)) for a in action * 3])
        print('Res_act', res_act)

        return res_act

    def update_rhs(self, visib, action):
        pass

    def reset(self):
        pass
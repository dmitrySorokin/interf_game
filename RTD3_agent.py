import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_h_size, n_latent_var, add_inp=0, encoder='standart', dropout=0,
                 device='cpu'):
        super(Actor, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.rnn_h_size = rnn_h_size
        # Changed Body
        if encoder == 'standart':
            self.body = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.Flatten()
            ).to(self.device)
        elif encoder == 'VGG':
            self.body = nn.Sequential(  # receive 16 x 64 x 64
                nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 x 64 x 64
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32 x 64 x 64
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 32 x 32
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64 x 32 x 32
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 x 16 x 16
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 x 16 x 16
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 x 8 x 8
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 x 8 x 8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            ).to(self.device)
        elif encoder == 'Standart_w_do':
            self.body = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.Flatten()
            ).to(self.device)

        self.body_out_shape = self._get_layer_out(self.body, state_dim)
        self.rnn = nn.GRU(self.body_out_shape + action_dim + add_inp, rnn_h_size, batch_first=True)

        self.nonlinearity = nn.Sequential(
            nn.Linear(rnn_h_size, n_latent_var),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        # mu head
        self.mu_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh()
        ).to(self.device)


    def _get_layer_out(self, layer, shape):
        o = layer(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.shape))

    def set_initial_rnn_state(self, batch_size):
        self.hx = torch.zeros((1, batch_size, self.rnn_h_size)).to(self.device)

    def forward(self, obs, p_act, masks=None, a_add=None):
        obs = torch.from_numpy(obs).float().to(self.device)
        # obs = (obs - obs.min()) / obs.max() # Min max scale
        obs /= 255
        if isinstance(p_act, np.ndarray):
            p_act = torch.from_numpy(p_act).float().to(self.device)
        if a_add is not None:
            a_add = torch.from_numpy(a_add).float().to(self.device)
            p_act = torch.cat((p_act, a_add), dim=-1)

        if masks is None:
            masks = torch.ones(p_act.shape[:-1]).to(self.device)

        hxs = self.hx
        # Let's figure out which steps in the sequence have a zero for any agent
        # We will always assume t=0 has a zero in it as that makes the logic cleaner
        has_zeros = ((masks[:, 1:] == 0.0) \
                     .any(dim=0)
                     .nonzero()
                     .squeeze()
                     .cpu())

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [int(obs.shape[1])]

        batch_size = obs.shape[0]
        obs = obs.view(-1, *self.state_dim)  # N * Seq_len x 16 x 64 x 64
        obs = self.body(obs)
        obs = obs.view(batch_size, -1, obs.shape[-1])  # N x Seq_len x 1024
        obs = torch.cat((obs, p_act), -1)

        outputs = []
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            out, hxs = self.rnn(obs[:, start_idx:end_idx],
                                hxs * masks[:, start_idx].view(1, -1, 1))

            outputs.append(out)

        x = torch.cat(outputs, dim=1)
        x = self.nonlinearity(x)

        mu = self.mu_layer(x)
        self.hx = hxs

        return mu


class Actor_v2(nn.Module):
    def __init__(self, state_dim, action_dim, rnn_h_size, n_latent_var, add_inp=0, encoder='standart', dropout=0,
                 device='cpu'):
        super(Actor_v2, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.rnn_h_size = rnn_h_size
        self.encoder = encoder
        # Changed Body
        if encoder == 'standart':
            self.body = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.Flatten()
            ).to(self.device)
        elif encoder == 'VGG':
            self.body = nn.Sequential(  # receive 16 x 64 x 64
                nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 x 64 x 64
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32 x 64 x 64
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 32 x 32
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64 x 32 x 32
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 x 16 x 16
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 x 16 x 16
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 x 8 x 8
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 x 8 x 8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 256 x 4 x 4
                nn.Flatten()
            ).to(self.device)
        elif encoder == 'Standart_w_dropout':
            self.body = nn.Sequential(  # receive 16 x 64 x 64
                nn.Conv2d(16, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Flatten()
            ).to(self.device)

        self.body_out_shape = self._get_layer_out(self.body, state_dim)
        #         self.rnn = nn.GRU(self.body_out_shape + action_dim + add_inp, rnn_h_size, batch_first=True)

        self.nonlinearity = nn.Sequential(
            #             nn.Linear(rnn_h_size, n_latent_var),
            nn.Linear(self.body_out_shape + action_dim + add_inp, n_latent_var),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_latent_var, rnn_h_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        self.rnn = nn.GRU(n_latent_var, rnn_h_size, batch_first=True)
        # mu head
        self.mu_layer = nn.Sequential(
            nn.Linear(rnn_h_size, action_dim),
            nn.Tanh()
        ).to(self.device)

    def _get_layer_out(self, layer, shape):
        o = layer(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.shape))

    def set_initial_rnn_state(self, batch_size):
        self.hx = torch.zeros((1, batch_size, self.rnn_h_size)).to(self.device)

    def get_initial_rnn_state(self, batch_size):
        return torch.zeros((batch_size, self.rnn_h_size)).to(self.device)

    def forward(self, obs, p_act, masks=None, a_add=None):
        obs = torch.from_numpy(obs).float().to(self.device)
        obs /= 255
        if isinstance(p_act, np.ndarray):
            p_act = torch.from_numpy(p_act).float().to(self.device)
        if a_add is not None:
            a_add = torch.from_numpy(a_add).float().to(self.device)
            p_act = torch.cat((p_act, a_add), dim=-1)

        if masks is None:
            masks = torch.ones(p_act.shape[:-1]).to(self.device)

        hxs = self.hx
        # Let's figure out which steps in the sequence have a zero for any agent
        # We will always assume t=0 has a zero in it as that makes the logic cleaner
        has_zeros = ((masks[:, 1:] == 0.0) \
                     .any(dim=0)
                     .nonzero()
                     .squeeze()
                     .cpu())

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [int(obs.shape[1])]

        batch_size = obs.shape[0]
        obs = obs.view(-1, *self.state_dim)  # N * Seq_len x 16 x 64 x 64
        obs = self.body(obs)
        obs = obs.view(batch_size, -1, obs.shape[-1])  # N x Seq_len x 1024
        obs = torch.cat((obs, p_act), -1)
        obs = self.nonlinearity(obs)

        outputs = []
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            out, hxs = self.rnn(obs[:, start_idx:end_idx],
                                hxs * masks[:, start_idx].view(1, -1, 1))

            outputs.append(out)

        x = torch.cat(outputs, dim=1)
        #         x = self.nonlinearity(x)

        mu = self.mu_layer(x)
        self.hx = hxs

        return mu


class TD3Agent:
    def __init__(self, model_path, observation_shape, action_dim, rnn_h_size, n_latent_var, encoder, add_inp=0):
        self.action_dim = action_dim

        print('in_channels = {}, action_dim = {}'.format(observation_shape, self.action_dim))

        td3_model = Actor_v2(observation_shape, self.action_dim, rnn_h_size, n_latent_var, add_inp, encoder)
        state = torch.load(model_path, map_location='cpu')
        if 'actor' in state:
            state = state['actor']
        td3_model.load_state_dict(state)
        td3_model.eval()

        self.td3_model = td3_model
        self.obs_shape = observation_shape
        self.p_act = np.zeros(self.action_dim)

    def get_action(self, state):
        # print('Hidden', self.td3_model.hx)
        # # print(state.shape)
        # cam_min_intense = np.min([np.sum(img) for img in state])
        # cam_max_intense = np.max([np.sum(img) for img in state])
        # # print([np.sum(img) for img in state])
        # camera_visib = (cam_max_intense - cam_min_intense) / (cam_max_intense + cam_min_intense)
        # mu = self.td3_model.forward(state[None, None, ...], self.p_act[None, None, ...],
        #                             a_add=camera_visib[None, None, None, ...])

        mu = self.td3_model.forward(state[None, None, ...], self.p_act[None, None, ...],
                                    a_add=None)
        action = mu.squeeze().detach().cpu().numpy()
        self.p_act = action
        res_act = np.array([0 if abs(a) < 0.5 else 10 ** (a-3) if a > 0 else -(10 ** (-a - 3)) for a in action * 3])
        print('Res_act', res_act)
        # print('Cam_vis', camera_visib)

        return res_act
        # return np.zeros(5)

    def update_rhs(self, visib, action):
        pass

    def reset(self):
        self.p_act = np.zeros(self.action_dim)
        self.td3_model.set_initial_rnn_state(1)
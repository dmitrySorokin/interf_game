import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, encoder='default', device='cpu'):
        super(Actor, self).__init__()
        self.device = device
        self.action_dim = action_dim
        # Changed Body
        if encoder == 'default':
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

        self.body_out_shape = self._get_layer_out(self.body, state_dim)

        self.nonlinearity = nn.Sequential(
            nn.Linear(self.body_out_shape, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU()
        ).to(self.device)

        # mu head
        self.mu_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh()
        ).to(self.device)

        self.outputs = dict()
        self.cX = 64 // 2
        self.cY = 64 // 2

        # Center finding
        self.cX = 32
        self.cY = 32

    def _get_layer_out(self, layer, shape):
        o = layer(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.shape))

    def forward(self, obs):
        # Center finding
        # coords = np.transpose(np.argwhere(obs > 0))
        # alpha = 0.8
        # self.cX = alpha * self.cX + (1 - alpha) * coords[2].mean()
        # self.cY = alpha * self.cY + (1 - alpha) * coords[3].mean()
        # print('Center x=', self.cX, 'center y=', self.cY)


        # print(obs.shape)
        # plt.show()
        #
        # plt.figure(1)
        # plt.hist(obs.flatten(), bins=100, range=(1, 255))
        # plt.show()

        # threshed_obs = []
        # for i in range(obs.shape[1]):
        #     o_i = obs[0, i] - obs[0, i].min()
        #     threshed_obs.append(o_i)
        # obs = np.array(threshed_obs)[None, ...]

        # OTSU + blur
        # threshed_obs = []
        # for i in range(obs.shape[1]):
        #     o_i = cv2.GaussianBlur(obs[0, i], (3,3), 0)
        #     # o_i = obs[0, i]
        #     _, o_i = cv2.threshold(o_i, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        #     threshed_obs.append(o_i)
        # obs = np.array(threshed_obs)[None, ...]

        # # print(obs)
        # plt.figure(2)
        # plt.hist(obs.flatten(), bins=100, range=(1, 255))
        # plt.show()
        # # plt.imshow(obs[0, 0])
        # # plt.show()
        # stop

        obs = obs / 255
        # obs = (obs - obs.min()) / obs.max()
        obs = torch.from_numpy(obs).float().to(self.device)

        obs = self.body(obs)
        obs = self.nonlinearity(obs)

        mu = self.mu_layer(obs)

        return mu

class TD3Agent:
    def __init__(self, model_path, observation_shape, action_dim, n_latent_var, encoder, action_rescale=True):
        self.action_dim = action_dim
        self.action_rescale = action_rescale

        print('in_channels = {}, action_dim = {}'.format(observation_shape, self.action_dim))

        td3_model = Actor(observation_shape, self.action_dim, n_latent_var, encoder)
        state = torch.load(model_path, map_location='cpu')
        if 'actor' in state:
            state = state['actor']
        td3_model.load_state_dict(state)
        td3_model.eval()

        self.td3_model = td3_model
        self.obs_shape = observation_shape

    def get_action(self, state):

        mu = self.td3_model.forward(np.array([state]))
        action = mu.detach().cpu().numpy()[0]
        if self.action_rescale:
            res_act = np.array([0 if abs(a) < 0.5 else 10 ** (a-3) if a > 0 else -(10 ** (-a - 3)) for a in action * 3])

        # print('Res_act', res_act)

        return res_act
        # return np.zeros(5)

    def update_rhs(self, visib, action):
        pass

    def reset(self):
        pass
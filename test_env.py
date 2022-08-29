import gym
from dqn_agent import DQNAgent
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import gym_interf

sns.set_theme()


def shift(state):
    start = np.random.randint(0, len(state))
    result = []
    for i in range(start, start + len(state)):
        result.append(state[i % len(state)])
    return np.asarray(result)


class BeamRadiusRandomizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._radius = env.radius

    def reset(self):
        r = self._radius * np.random.uniform(0.8, 1.2)
        self.env.set_radius(r)
        return self.env.reset()


class ChannelShifter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return shift(self.env.reset())

    def step(self, actions):
        n_back = np.random.randint(0, 8)
        self.env.set_backward_frames(n_back)
        obs, rew, done, info = self.env.step(actions)
        return shift(obs), rew, done, info


class BrightnessRandomizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def randomize(self, obs):
        obs = obs * np.random.uniform(0.7, 1.3)
        obs = np.minimum(obs, 255)
        return obs.astype(np.uint8)

    def reset(self):
        obs = self.env.reset()
        return self.randomize(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.randomize(obs), rew, done, info


class StepRandomizer(gym.Wrapper):
    def step(self, action):
        return self.env.step(action * np.random.uniform(0.96, 1.04))


def make_env(env_version):
    env = gym.make(f'interf-v{env_version}')
    env = BeamRadiusRandomizer(env)
    env = BrightnessRandomizer(env)
    env = ChannelShifter(env)
    env = StepRandomizer(env)
    return env


def plot(xdata, ydata, fig, label='', title='', xlabel='', ylabel='', savename=''):
    mean = np.mean(ydata, axis=0)
    std = np.std(ydata, axis=0)
    plt.plot(xdata, mean, label=label)
    plt.fill_between(xdata, mean - std, mean + std, alpha=0.5)
    
    plt.title(title)
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(savename, bbox_inches='tight', format='pdf')


def test(env, agent, env_name, n_games=100):
    visib, visib_device = [], []
    for game_id in trange(n_games):
        game_visib, game_visib_device = [], []
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            game_visib.append(info['visib'])
            game_visib_device.append(info['visib_device'])
        visib.append(game_visib)
        visib_device.append(game_visib_device)

    visib = np.array(visib)
    visib_device = np.array(visib_device)
    visib_diff = np.abs(visib - visib_device)
    steps = range(len(visib[0]))

    fig1 = plt.figure(1)
    plot(steps, visib_diff, fig1, label=env_name, title='MAE(Visibility)', xlabel='Steps', ylabel='MAE', savename='visib_diff.pdf')

    fig2 = plt.figure(2)
    plot(steps, visib, fig2, label=env_name, title='Aligning with DQN', xlabel='Steps', ylabel='Visibility', savename='test_env.pdf')

    vis_camera_mean = np.mean(visib, axis=0)
    vis_camera_std = np.std(visib, axis=0)
    print(f'visib: {vis_camera_mean[-1]} +- {vis_camera_std[-1]}; min = {np.min(visib, axis=0)[-1]}')
    print(f'visib_device: {np.mean(visib_device, axis=0)[-1]} +- {np.std(visib_device, axis=0)[-1]}')


if __name__ == '__main__':
    models = ['ablation_models/all_random', 'lense_models/all_random_r_714', 'two_telescope_models/all_random']

    for model, version in zip(models, range(1, 4)):
        env = make_env(version)
        env.use_beam_masks(False)

        agent = DQNAgent(
            model,
            env.observation_space.shape,
            env.action_space.shape,
            step_fractions=(0.01, 0.05, 0.1)
        )

        print(f'env = interf-v{version}, model = {model}')
        test(env, agent, f'interf-v{version}')
        env.close()

import gym
import gym_interf
from dqn_agent import DQNAgent
from a2c_agent import A2CAgent
from tqdm import trange

env = gym.make('interf-v1')
print(env.observation_space)
exit(0)


#type(env).n_points = 256
#type(env).x_min = -4
#type(env).x_max = 4
max_steps = type(env).max_steps

agent = DQNAgent('models/dqn_v1', env.observation_space.shape[0], env.action_space.n)
#agent = A2CAgent('models/a2c_interf_model_v2', env.observation_space.shape, env.action_space.n)


s = env.reset()
n_games = 0
n_steps = 0
n_solved = 0
while n_games < 10:
    a = agent.get_action(s)
    n_steps += 1
    s, r, done, info = env.step(a)
    if done:
        s = env.reset()
        print('n_steps = {}, visib = {}'.format(n_steps, info['visib']))
        n_solved += n_steps < max_steps
        n_steps = 0
        n_games += 1

print('solved {} games out of {}'.format(n_solved, n_games))


def evaluate(n_games=10000):
    n_solved = 0
    n_steps = 0
    dist = 0
    angle = 0
    for i in trange(n_games):
        s = env.reset()
        istep = 0
        while(True):
            a = agent.get_action(s)
            s, reward, done, info = env.step(a)
            istep += 1
            if done:
                n_solved += istep < 200
                n_steps += istep
                dist += info['dist']
                angle += info['angle_between_beams']
                break
    return n_solved / n_games, n_steps / n_games, dist / n_games, angle / n_games

print(evaluate())

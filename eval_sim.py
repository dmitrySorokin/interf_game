import gym
import gym_interf
from dqn_agent import DQNAgent
import numpy as np
from tqdm import trange
from wrappers import DiscreteActionWrapper
import os
import argparse


env = DiscreteActionWrapper(gym.make('interf-v1'), 1)
env.use_beam_masks(False)
env.set_radius(0.957)


parser = argparse.ArgumentParser()
parser.add_argument(
        '--model',
        type=str,
        help='path to trained model.\nTested: ablation_models/all_random, \
            ablation_models/no_radius_random, \
            ablation_models_no_brightness_random, \
            ablation_models/no_noise, \
            ablation_models/no_channel_shift',
        default='ablation_models/all_random')
parser.add_argument('--log_dir', type=str, help='path to log dir', default='all_random')
parser.add_argument('--ngames', type=int, help='number of episodes', default=100)


#agent = DQNAgent('eval_models/dqn_exp_model', (16, 64, 64), step_fractions=(0.05,))
#agent = DQNAgent('eval_models/dqn_exp_model_diff_actions', (16, 64, 64))
#agent = DQNAgent('eval_models/dqn_exp_log_loss', (16, 64, 64), step_fractions=(0.05,))
#agent = DQNAgentWithHistory('eval_models/dqn_diff_actions_lstm', (16, 64, 64))

args = parser.parse_args()
agent = DQNAgent(args.model, env.observation_space.shape, env.action_space.shape, step_fractions=(0.01, 0.05, 0.1))


max_steps = 100
ngames = args.ngames 
exp_name = args.log_dir

os.mkdir(exp_name)
with open('{}/log.txt'.format(exp_name), 'w') as f:
    f.write('igame;istep;visib_camera;visib_device\n')
    for igame in trange(ngames):
        print('play game #', igame)
        done = False
        state = env.reset()
        agent.reset()
        for istep in range(max_steps):
            action = agent.get_action(state)
            discrete_action = agent.get_discrete_action(state, 0)
            next_state, reward, done, info = env.step([action, 'qrobot'])
            visib = info['visib']
            agent.update_rhs(visib, action)
            center = info['proj_2']
            kvector = info['kvector']

            f.write('{};{};{};{}\n'.format(igame, istep, visib, visib))
            np.savez(
                '{}/game_{}_step_{}'.format(exp_name, igame, istep),
                state=state,
                action=discrete_action,
                next_state=next_state,
                done=done,
                visib_device=visib,
                visib_camera=visib,
                center=center,
                kvector=kvector
            )

            state = next_state

            if done:
                break

        print('============GAME OVER=================')


env.close()

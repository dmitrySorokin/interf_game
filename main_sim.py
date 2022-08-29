import gym
import gym_interf
from gym_play import play, PlayPlot
from dqn_agent import DQNAgent
# from CCPPO_agent import CCPPOAgent
from TD3_agent import TD3Agent
from wrappers import DiscreteActionWrapper
import argparse


IMAGE_SIZE = 64
ZOOM_1_IMAGE_SIZE = 512
ZOOM = ZOOM_1_IMAGE_SIZE // IMAGE_SIZE

gym_interf.InterfEnv.n_points = IMAGE_SIZE
gym_interf.InterfEnv.max_steps = 1e6
#gym_interf.InterfEnv.n_frames = 16
#gym_interf.InterfEnv.done_visibility = 0.95


parser = argparse.ArgumentParser()
parser.add_argument(
        '--model', 
        type=str, 
        help='path to trained model. Tested: ablation_models/all_random, \
            ablation_models/no_radius_random, \
            ablation_models_no_brightness_random, \
            ablation_models/no_noise, \
            ablation_models/no_channel_shift', 
        default='ablation_models/all_random')
parser.add_argument(
    '--version',
    type=str,
    help='version of the env. v1 - interf without lenses, v2 - interf with lenses',
    default='v1',
    choices={'v1', 'v2', 'v3'})
parser.add_argument(
    '--agent',
    type=str,
    help='agent to run',
    choices={'dqn', 'td3'},
    default='dqn')

args = parser.parse_args()

env = DiscreteActionWrapper(gym.make(f'interf-{args.version}'), 1)

#env.set_calc_image('gpu')
env.add_noise(0)
env.use_beam_masks(False)
#env.set_backward_frames(2)
env.set_radius(0.714)

# Tested agents
#agent = DQNAgent('eval_models/dqn_exp_model_diff_actions', (16, 64, 64))
#agent = DQNAgent('eval_models/dqn_exp_model', (16, 64, 64), step_fractions=(0.05, ))
#agent = DQNAgent('eval_models/dqn_exp_log_loss_diff_actions', (16, 64, 64))
#agent = DQNAgent('ablation_models/no_noise', (16, 64, 64), step_fractions=(0.01, 0.05, 0.1))


# Not tested agents
#agent = A2CAgent('models/a2c_interf_model', env.observation_space.shape, 8)
#agent = PPOAgent('models/experiment_ppo_change_camera_v3', (16, 64, 64), env.action_space)
#agent = DoubleSetAgent('models/prev_state_model', (32, 64, 64), env.action_space)
#agent = DoubleSetRegressor('models/regressor_model2', (32, 64, 64), env.action_space)
#agent = PPOAgent('eval_models/ppo_change_camera_v2', (16, 64, 64), env.action_space)

if args.agent == 'dqn':
    agent = DQNAgent(
        args.model,
        env.observation_space.shape,
        env.action_space.shape,
        step_fractions=(0.01, 0.05, 0.1)
    )
elif args.agent == 'td3':
    agent = TD3Agent('./lense_models/agent_lense_TD3_VGG_randoms_wo_pos_.714_camera_visib.pt', (16,64,64), 5, 512, 'VGG')
else:
    raise ValueError(f'unsupported agent {args.model}')

#agent = CCPPOAgent(args.model, (16, 64, 64), 4, 256)
print(agent)

def callback(obs_t, obs_tp1, action, rew, done, info):
    # print('fit time', info['fit_time'])
    print('state calc time', info['state_calc_time'])
    #print('proj1 = {}, proj2 = {}'.format(info['proj_1'], info['proj_2']))
    #print('mirror1_normal = {}, mirror2_normal = {}'.format(info['mirror1_normal'], info['mirror2_normal']))
    #print('imin = {}, imax = {}'.format(info['imin'], info['imax']))

    if done:
        print('GAME OVER', 'visib = ', info['visib'])
    return [
        info['visib'],
        info['dist'],
        info['angle_between_beams'],
        info['r_curvature'],
        info['radius_bottom']
    ]


plotter = PlayPlot(callback, 30 * 5, [
    lambda info: 'visib = {:.4f} / {:.4f}'.format(info['visib'], info['visib_device']),
    lambda info: 'dist = {:.2f}'.format(info['dist']),
    lambda info: 'angle_between = {:.2f}'.format(info['angle_between_beams']),
    lambda info: 'r_curvature = {:.2f}'.format(info['r_curvature']),
    lambda info: 'raidus_bottom = {:.2f}'.format(info['radius_bottom']),
])

play(env=env, zoom=8, fps=10, plotter=plotter, agent=agent)

import gym
import iron_interf
from gym_play import play, PlayPlot
from dqn_agent import DQNAgent
#from a2c_agent import A2CAgent
#from ppo_agent import PPOAgent
from TD3_agent import TD3Agent
import numpy as np
import time
from wrappers import DiscreteActionWrapper


iron_interf.IronInterfEnv.n_points = 1024
#iron_interf.IronInterfEnv.max_steps = 1e6
#iron_interf.IronInterfEnv.done_visibility = 2.0


def curr_time():
    tm = time.localtime()
    res = '{}_{}_{}_{}_{}'.format(tm.tm_mon, tm.tm_wday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    return res


def to_agent_action(actions):
    actions = np.array(actions) * 20
    if np.sum(actions == np.array([-1, 0, 0, 0])) == 4:
        return 0
    elif np.sum(actions == np.array([1, 0, 0, 0])) == 4:
        return 1
    elif np.sum(actions == np.array([0, -1, 0, 0])) == 4:
        return 2
    elif np.sum(actions == np.array([0, 1, 0, 0])) == 4:
        return 3
    elif np.sum(actions == np.array([0, 0, -1, 0])) == 4:
        return 4
    elif np.sum(actions == np.array([0, 0, 1, 0])) == 4:
        return 5
    elif np.sum(actions == np.array([0, 0, 0, -1])) == 4:
        return 6
    elif np.sum(actions == np.array([0, 0, 0, 1])) == 4:
        return 7
    elif np.sum(actions == np.array([0, 0, 0, 0])) == 4:
        return 8
    assert False, actions


env = DiscreteActionWrapper(gym.make('iron_interf-v2'), 1)
env.enable_camera(True)
env.set_exposure(1)
#env = DiscreteActionWrapper(env)
#env.set_calc_image('gpu')
agent = DQNAgent('lense_models/all_random_r_714', (16, 64, 64), (5,)) # THIS IS PROD UNCOMMENT ME


#agent = DQNAgent('eval_models/dqn_exp_model', (16, 64, 64), step_fractions=(0.05,))
#agent = DQNAgent('eval_models/dqn_exp_model_diff_actions', (16, 64, 64))
#agent = DQNAgent('eval_models/dqn_exp_log_loss_diff_actions', (16, 64, 64), (4,)) # THIS IS PROD UNCOMMENT ME
#agent = DQNAgent('ablation_models/no_noise', (16, 64, 64), step_fractions=(0.01, 0.05, 0.1))
#agent = TD3Agent('cont_models/agent_TD3_VGG_all_random_punish.pt', (16,64,64), 4, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_.714.pt', (16,64,64), 5, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_.714_camera_visib.pt', (16,64,64), 5, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_min_max_scaler.pt', (16,64,64), 5, 512, 'VGG')



#agent = PPOAgent('models/ppo_model', env.observation_space.shape, env.action_space)
#agent = PPOAgent('models/ppo_change_camera_v2', env.observation_space.shape, env.action_space)


def callback(obs_t, obs_tp1, action, rew, done, info):
    # print('fit time', info['fit_time'])
    print('state calc time', info['state_calc_time'])
    # print('proj1 = {}, proj2 = {}'.format(info['proj_1'], info['proj_2']))
    # print('mirror1_normal = {}, mirror2_normal = {}'.format(info['mirror1_normal'], info['mirror2_normal']))
    # print('imin = {}, imax = {}'.format(info['imin'], info['imax']))

    #act = to_agent_action(action[0])
    #np.savez('data2/' + curr_time(), state=obs_t, next_state=obs_tp1, action=act, reward=rew, done=done)

    if done:
        print('GAME OVER', 'visib = ', info['visib_camera'])
    return [
        info['visib_camera'],
        info['visib_device'],
        info['tot_intens_device']
        #info['dist'],
        #info['angle_between_beams']
    ]


def multipont_callback(obs_t, obs_tp1, action, rew, done, info):
    return info['tot_intens_device']


plotter = PlayPlot(
    callback, 30 * 5, [
        lambda info: 'visib_camera = {}'.format(info['visib_camera']),
        lambda info: 'visib_device = {}'.format(info['visib_device']),
    ],
    multipont_callback
)

play(env=env, zoom=8, fps=20, plotter=plotter, agent=agent)

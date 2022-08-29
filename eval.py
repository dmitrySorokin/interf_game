import gym
import iron_interf
from dqn_agent import DQNAgent
from dqn_agent_with_history import DQNAgentWithHistory
import numpy as np
from tqdm import trange
from wrappers import DiscreteActionWrapper
import os
from TD3_agent import TD3Agent
from RTD3_agent import TD3Agent as RTD3Agent


env = DiscreteActionWrapper(gym.make('iron_interf-v2'), 1)
env.enable_camera(True)
env.set_exposure(1)


#agent = DQNAgent('eval_models/dqn_exp_model', (16, 64, 64), step_fractions=(0.05,))
# agent = DQNAgent('eval_models/dqn_exp_model_diff_actions', (16, 64, 64))
#agent = DQNAgent('eval_models/dqn_exp_log_loss_diff_actions', (16, 64, 64))
#agent = DQNAgent('eval_models/dqn_exp_log_loss', (16, 64, 64), step_fractions=(0.05,))
#agent = DQNAgentWithHistory('eval_models/dqn_diff_actions_lstm', (16, 64, 64))
#agent = DQNAgent('ablation_models/no_noise', (16, 64, 64), step_fractions=(0.01, 0.05, 0.1))
# agent = TD3Agent('cont_models/agent_TD3_VGG_all_random_punish.pt', (16,64,64), 4, 512, 'VGG')
# agent = DQNAgent('lense_models/all_random_r_714', (16, 64, 64), (5,)) # THIS IS PROD UNCOMMENT ME
# agent = DQNAgent('lense_models/all_random_r_714_lens_step', (16, 64, 64), (5,)) # fixed lens step
# agent = DQNAgent('lense_models/agent_DQN_piezo_noise_0.7_lense_step233.pt', (16, 64, 64), (5,)) # fixed lens step
# Without action rescale
agent = TD3Agent('lense_models/agent_lense_TD3_VGG_random_wo_pos_no_action_rescaler259.pt', (16,64,64), 5, 512, 'VGG', False)


# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos.pt', (16,64,64), 5, 512, 'VGG') # Wrong radius
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_.714_camera_visib.pt', (16,64,64), 5, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_big_reward.pt', (16,64,64), 5, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_min_max_scaler.pt', (16,64,64), 5, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_.714.pt', (16,64,64), 5, 512, 'VGG')

# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_0.7_lense_step247.pt', (16,64,64), 5, 512, 'VGG')
# agent = TD3Agent('lense_models/agent_lense_TD3_VGG_randoms_wo_pos_piezo_noise_dev_vis_0.7_lense_step337.pt', (16,64,64), 5, 512, 'VGG') # Best TD3

# agent = RTD3Agent('./lense_models/agent_lense_RTD3_VGG_randoms_wo_pos_piezo_noise_dev_vis_0.7_lense_step278.pt', (16,64,64), 5, 512, 512, 'VGG', 0)

max_steps = 100
ngames = 50
new_log = True
game_start_id = 0
# exp_name = 'lense_TD3_eval_fixed_lense_step_fixed_fps' # Best TD3
# exp_name = 'lense_TD3_eval_fixed_lense_step_fixed_fps_piezo_noise' # piezo TD3

# exp_name = 'lense_DQN_eval_fixed_fps' # DQN
# exp_name = 'lense_DQN_eval_fixed_lense_step_fixed_fps' # fixed lense step DQN
# exp_name = 'lense_DQN_eval_fixed_lense_step_fixed_fps_piezo_noise' # Stepan's fixed lense step, piezo_noise DQN

# Without action rescale
exp_name = 'lense_TD3_eval_no_action_rescale_fixed_lense_step_fixed_fps_piezo_noise'

# exp_name = 'lense_RTD3_eval_fixed_lense_step_fixed_fps_piezo_noise' # piezo RTD3

try:
    os.mkdir(exp_name)
except FileExistsError:
    print('Directory is already exist')
    new_log = False
    with open('{}/log.txt'.format(exp_name), 'r') as f:
        last_line = "0;0;0;0"
        for line in f:
            if len(line.split(';')) == 4:
                last_line = line
    game_start_id = int(last_line.split(';')[0]) + 1
    print('Starting index = ', game_start_id)

with open('{}/log.txt'.format(exp_name), 'a') as f:
    if new_log:
        f.write('igame;istep;visib_camera;visib_device\n')
    for igame in trange(game_start_id, game_start_id + ngames):
        print('play game #', igame)
        done = False
        state = env.reset()
        agent.reset()
        for istep in range(max_steps):
            action = agent.get_action(state)
            #discrete_action = agent.get_discrete_action(state, 0)
            next_state, reward, done, info = env.step([action, 'qrobot'])
            visib_camera = info['visib_camera']
            visib_device = info['visib_device']

            agent.update_rhs(visib_device, action)

            print('VISIB = ', visib_camera)
            f.write('{};{};{};{}\n'.format(igame, istep, visib_camera, visib_device))
            np.savez(
                '{}/game_{}_step_{}'.format(exp_name, igame, istep),
                state=state,
                action=action,
                next_state=next_state,
                done=done,
                visib_device=visib_device,
                visib_camera=visib_camera
            )

            state = next_state

            # if info['imax'] < 350:
            #     val = input('imax = {}; press y to continue; press n to quit'.format(info['imax']))
            #     if val == 'n':
            #         env.close()
            #         exit(0)

            if visib_camera > 0.9:
                print('VISIB IS GOOD: RESET MIRROR POSITIONS')
                env.reset_mirror_positions()

            if done:
                break

        print('============GAME OVER=================')
        print('VISIB = ', visib_camera)
        #if visib_camera < 0.5:
        #    val = input('VISIB IS LOW; press y to continue; press n to quit'.format(info['imax']))
        #    if val == 'n':
        #        env.close()
        #        exit(0)


env.close()

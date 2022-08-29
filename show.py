#!/usr/bin/python3

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import argparse

from dqn_agent import DQNAgent
matplotlib.use('TkAgg')

def to_cont_action(action_id: int):
    step_fractions = [0.01, 0.05, 0.1]
    actions = [0, 0, 0, 0]
    if action_id < 24:

        step_fraction = step_fractions[action_id // 8]
        action_id = action_id % 8
        actions[action_id // 2] = step_fraction * (
            -1 if action_id % 2 == 0 else 1
        )

    actions = [actions[1], actions[0], actions[3], actions[2]]
    actions = np.asarray(actions)
    return actions


N_STEPS = 100
N_FRAMES = 16


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--game_id', type=int, default=0)
parser.add_argument('--save_video', type=bool, default=False)
parser.add_argument('--video_name', type=str, default='im.mp4')

args = parser.parse_args()

print('open folder = {}, game = {}'.format(args.folder, args.game_id))
#agent = DQNAgent('eval_models/dqn_exp_log_loss_diff_actions', (16, 64, 64))
<<<<<<< Updated upstream
agent = DQNAgent(
    'lense_models/all_random',
    (16, 64, 64),
    (5,),
    step_fractions=(0.01, 0.05, 0.1)
)

=======
>>>>>>> Stashed changes

fig, ax = plt.subplots()
fig.set_facecolor("grey")

ax.set_axis_off()
im = ax.imshow(np.zeros((64, 64)), animated=True, vmin=0, vmax=255, cmap='gray')


def foo(iframe):
    iframe = iframe % (N_STEPS * N_FRAMES)
    istep = iframe // N_FRAMES
    loaded = np.load(os.path.join(args.folder, 'game_{}_step_{}.npz'.format(args.game_id, istep)))
    state = loaded['state']    	
    #q_vals = agent.get_q_values(state)[0]

    #qval_ids = np.argsort(-q_vals)
    #for qval_id in qval_ids:
    #    print(
    #        'agent old: action({}) = {}; qvalue = {};'.format(
    #            qval_id, agent._to_cont_action(qval_id), q_vals[qval_id],
    #        )
    #    )
    #print('=' * 30)

    iimg = iframe % N_FRAMES
    img = state[iimg]
<<<<<<< Updated upstream
   
    # FIXME add actions to eval data
    #action = to_cont_action(loaded['action'])
    #action = ', '.join(map(str, action))
    action ='unknown'

    plt.title('step = {}; visib = {:.2f}; action = [{}]'.format(istep, loaded['visib_camera'], action)) 
=======
    # We don't write 'action' in npz file
    # plt.title('step = {}; visib = {:.2f}; action = {}'.format(istep, loaded['visib_camera'], loaded['action']))
    plt.title('step = {}; camera_visib = {:.2f}; device_visib = {:.2f};'.format(istep, loaded['visib_camera'], loaded['visib_device']))
>>>>>>> Stashed changes
    im.set_array(img)


ani = animation.FuncAnimation(fig, foo, interval=80, save_count=N_STEPS * N_FRAMES)

if args.save_video:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='RQC'), bitrate=1800)
    ani.save(args.video_name, writer=writer)
else:
    plt.show()


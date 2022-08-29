import gym
import pygame
import matplotlib
from gym import logger
import time
import cv2
from scipy.optimize import curve_fit
import os
import numpy as np


try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None

from collections import deque
from pygame.locals import VIDEORESIZE
import numpy as np


def to_gray_scale(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def display_arr(screen, arr, video_size, transpose):
    image = to_gray_scale(arr)
    pyg_img = pygame.surfarray.make_surface(image.swapaxes(0, 1) if transpose else image)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def curr_time():
    tm = time.localtime()
    res = '{}_{}_{}'.format(tm.tm_hour, tm.tm_min, tm.tm_sec)
    return res


def save_image(image, image_name=None, resolution=(64, 64)):
    if image_name is None:
        image_name = curr_time() + '.png'

    image = cv2.cv2.resize(image, (resolution[0], resolution[1]))
    w, h = image.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image
    cv2.cv2.imwrite('images/{}'.format(image_name), ret)


def play(env, transpose=True, fps=30, zoom=None, plotter=None, keys_to_action=None, agent=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    noop = np.zeros(env.action_space.shape[0])
    env.reset(noop)
    agent.reset()
    rendered = env.render()

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = np.array([rendered.shape[2], rendered.shape[1]])
    if zoom is not None:
        video_size *= zoom

    action = None
    action_time = None
    running = True
    env_done = False
    agent_enabled = False

    pygame.init()
    icon = pygame.image.load('icon.jpg')
    pygame.display.set_caption('interferometer')
    pygame.display.set_icon(icon)
    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    obs = env.reset(noop)
    agent.reset()

    istep = 0
    igame = 0

    eval_name = None
    eval_log = None

    while running:
        if action is not None:
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            step_time = time.time()
            plotter.callback(prev_obs, obs, action, rew, env_done, info)
            print('game = {}; step = {}; action ({}) {}'.format(igame, istep, action[1], action[0]))
            istep += 1
            try:
                visib_camera = info['visib_camera']
                visib_device = info['visib_device']
            except KeyError:
                visib_camera = info['visib']
                visib_device = info['visib']

            if eval_name is not None:
                eval_log.write('{};{};{};{};{};{}\n'.format(igame, istep, visib_camera, visib_device, action_time, step_time))
                eval_log.flush()
                np.savez(
                    '{}/game_{}_step_{}'.format(eval_name, igame, istep),
                    state=prev_obs,
                    action=action[0],
                    step_fraction=env.step_fraction,
                    next_state=obs,
                    done=env_done,
                    visib_device=visib_device,
                    visib_camera=visib_camera
                )
            if agent_enabled:
                if env_done:
                    agent.reset()
                else:
                    agent.update_rhs(env.visib, action[0])

            if env_done:
                obs = env.reset(noop)
                env_done = False
                istep = 0
                igame += 1

        # TODO revert changes
        # play.frames_to_display = obs if not hasattr(play, 'frames_to_display') else obs
        # if (action is not None and all(sum(abs(np.array([action[0]])))) != 0):
        #     play.frames_to_display = obs
        action = None
        action_time = None
        for frame in env.render():
            if action is not None:
                break
            display_arr(screen, frame, transpose=transpose, video_size=video_size)
            pygame.display.flip()
            clock.tick(fps)

            for event in pygame.event.get():
                # test events, set key states
                if event.type == pygame.KEYDOWN:
                    if event.key in relevant_keys:
                        action = (keys_to_action.get(tuple(sorted([event.key])), 0), 'user')
                        action_time = time.time()
                    elif event.key == 27:
                        running = False
                    elif event.key == ord(' '):
                        agent_enabled = not agent_enabled
                    elif event.key == ord('q'):
                        env_done = False
                        agent_enabled = False
                        obs = env.reset(noop)
                        plotter.reset()
                        env.set_step_fraction(0.1)
                        istep = 0
                        igame += 1
                    elif event.key == ord('r'):
                        env_done = False
                        agent_enabled = False
                        obs = env.reset()
                        agent.reset()
                        plotter.reset()
                        env.set_step_fraction(0.1)
                        istep = 0
                        igame += 1
                    elif event.key == ord('p'):
                        save_image(frame)
                    elif event.key == ord('o'):
                        print('saving state...')
                        env.save_state('saved_states/' + curr_time())
                    elif event.key == ord('z'):
                        print('ZZZZ', event.key)
                        try:
                            str_val = input('insert new step_fraction: ')
                            step_fraction = float(str_val)
                            env.set_step_fraction(step_fraction)
                        except e:
                            print(e)
                    elif event.key == ord('1'):
                        env.set_step_fraction(0.01)
                    elif event.key == ord('2'):
                        env.set_step_fraction(0.05)
                    elif event.key == ord('3'):
                        env.set_step_fraction(0.1)
                    elif event.key == ord('e'):
                        eval_name = input('PLEASE ENTER YOUR NAME\n')
                        os.mkdir(eval_name)
                        eval_log = open('{}/log.txt'.format(eval_name), 'w')
                        eval_log.write('igame;istep;visib_camera;visib_device;action_time;step_time\n')
                    elif event.key == ord('0'):
                        action = (noop, 'qrobot')
                        action_time = time.time()
                    else:
                        print('UNKNOWN', event.key)

                elif event.type == pygame.QUIT:
                    running = False
                elif event.type == VIDEORESIZE:
                    video_size = event.size
                    screen = pygame.display.set_mode(video_size)
                    print(video_size)

        if agent_enabled and not env_done:
            action = (agent.get_action(obs), 'qrobot')
            action_time = time.time()

    env.close()
    pygame.quit()


class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names, multipoint_callback=None):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names
        self.multipoint_data = multipoint_callback

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots + (multipoint_callback is not None))
        if num_plots == 1:
            self.ax = [self.ax]
        #for axis, name in zip(self.ax, plot_names):
        #    axis.set_title(name)
        plt.tight_layout()
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
            self.ax[i].set_title(self.plot_names[i](info))
            self.ax[i].set_xlim(xmin, xmax)

        if self.multipoint_data:
            self.ax[-1].clear()

            mpoints = np.array(self.multipoint_data(obs_t, obs_tp1, action, rew, done, info))
            mrange = np.array(range(len(mpoints)))

            start_point = 40
            ydata = mpoints[start_point:]
            xdata = mrange[start_point:] - start_point

            self.ax[-1].scatter(xdata, ydata, c='blue')

            second_sin_start = info['second_sin_start'] - 3 - start_point
            #self.ax[-1].scatter([second_sin_start], [300], color='green')
            print(second_sin_start)

            def fit_foo(x, mean, ampl, omega1, phi11, phi12, phi2):
                use_first_sin1 = x < second_sin_start - 8
                use_first_sin2 = x > second_sin_start
                use_second_sin = 1 - (use_first_sin1 + use_first_sin2)

                return mean + ampl * (
                    np.sin(omega1 * x + phi11) * use_first_sin1 +
                    np.sin(omega1 * x + phi12) * use_first_sin2 +
                    np.sin(omega1 * 5 * x + phi2) * use_second_sin
                )

            try:
                popt, pcov = curve_fit(
                    fit_foo, xdata, ydata,
                    p0=(275, 130, 0.1, 0, 0, 0)
                    #bounds=(
                    #    (200, 80, 0.05, 0, 0, 40, 60, 0),
                    #    (350, 180, 0.2, 2 * np.pi, 2 * np.pi, 60, 80, 2 * np.pi)
                    #)
                    #bounds=(
                    #    (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 40, 40, -np.inf),
                    #    (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 80, 80, np.inf),
                    #)
                )

                #print('POPT: ', popt)
                #print('PCOV: ', pcov)

                self.ax[-1].plot(xdata, [fit_foo(val, *popt) for val in xdata], color='red')

                visib = np.abs(popt[1]) / popt[0]

                self.ax[-1].set_title('visib_fit = {}'.format(visib))
            except RuntimeError:
                print('CAN NOT FIT')

        plt.pause(0.000001)

    def reset(self):
        self.t = 0
        for d in self.data:
            d.clear()

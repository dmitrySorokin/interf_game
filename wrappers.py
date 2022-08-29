import gym
import numpy as np
import cv2
from collections import deque


class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, e, k):
        super().__init__(e)
        self.image_id = 0
        self.step_fraction = 0.1
        self.k = k
        self.shape = (k * 16, 64, 64)
        self.frames = deque([], maxlen=self.k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.asarray(self.frames).reshape(self.shape)

    def reset(self, actions=None):
        ob = self.env.reset(actions)
        #ob = self._shift(ob)
        for _ in range(self.k):
            self.frames.append(self._resize(ob))
        return self._get_ob()

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'human':
            return self.env.render(mode, **kwargs)
        state = self.env.render(mode, **kwargs)
        return self._resize(state)

    def _resize(self, images):
        n_imgs = images.shape[0]
        result = np.zeros(shape=(n_imgs, 64, 64), dtype=np.uint8)
        for i, img in enumerate(images):
            img = cv2.cv2.resize(img, (64, 64))
            result[i] = img
        return result

    def _shift(self, state):
        start = np.random.randint(0, len(state))
        result = []
        for i in range(start, start + len(state)):
            result.append(state[i % len(state)])
        return np.asarray(result)

    def set_step_fraction(self, value):
        self.step_fraction = value
        print('set step fraction to ', value)

    def step(self, action_with_source):
        if action_with_source[1] == 'qrobot':
            actions = action_with_source[0]
            #print(actions)
        else:
            action_id = action_with_source[0]
            actions = [0] * self.action_space.shape[0]
            actions[action_id // 2] = self.step_fraction * (
                    -1 if action_id % 2 == 0 else 1
            )

        state, reward, done, info = self.env.step(actions)

        #state = self._shift(state)
        self.frames.append(self._resize(state))

        return self._get_ob(), reward, done, info

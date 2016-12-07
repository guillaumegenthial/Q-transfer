"""
https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

"""
Modifications of the original env to create similar and related tasks by changing
    - dynamics
    - reward

Use :
    - set_reward_modes : to change reward as a combination of different rewards
    - set_slope : to change the slope of the problem, the higher, the more difficult
    - set_max_speed : to change max speed
    - set_power : to change the acceleration of the car, the higher, the easier

# Example
    ```python
    env.set_task(
        modes=[
        ("time", 1),
        ("energy", 0),
        ("distance", 0),
        ("center", 0),
        ("height", 0),
        ("still", 0)
        ], 
        slope=00025, 
        speed=0.07, 
        power=0.001, 
        min_position=-1.2,
        low=-0.6,
        high=-0.4,
        obstacles=[
        (-.5, .1, .01), 
        (0, .1, .05)], 
        actions_nb=3,
        neutral=1
        )
    ```
"""

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        # ------------ MODIFIED ------------
        self.set_task()
        # -------------------------------

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(self.actions_nb)
        self.observation_space = spaces.Box(self.low, self.high)


        self._seed()
        self.reset()

     


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # ------------  MODIFIED  ------------
    def set_task(
        self, 
        modes=[
        ("time", 1),
        ("energy", 0),
        ("distance", 0),
        ("center", 0),
        ("height", 0),
        ("still", 0)
        ], 
        slope=00025, 
        speed=0.07, 
        power=0.001, 
        min_position=-1.2,
        low=-0.6,
        high=-0.4,
        obstacles=[
        (-.5, .1, .01), 
        (0, .1, .05)], 
        actions_nb=3,
        neutral=1
        ):

        self.set_actions(actions_nb, neutral)
        self.set_reward_modes(modes)
        self.set_slope(slope)
        self.set_max_speed(speed)
        self.set_power(power)
        self.set_min_pos(min_position)
        self.set_reset_param(low, high)
        self.add_obstacles(obstacles)
        

    def set_actions(self, actions_nb=3, neutral=1):
        self.actions_nb = actions_nb
        self.neutral = neutral

    def set_reward_modes(self, modes=[("time", 1), ]):
        """
        list of tuples (mode, proportion)
        reward will be a weighted sum of the mode rewards
        """
        self.reward_modes = modes

    def set_slope(self, slope=0.0025):
        """
        if slope is higher, more difficult
        for slope = 0.0015, very easy (almost no strategy)
        """
        self.slope = slope

    def set_max_speed(self, speed=0.07):
        self.max_speed = speed
        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])
        self.observation_space = spaces.Box(self.low, self.high)

    def set_power(self, power=0.001):
        """
        acceleration of the car, higher is better
        """
        self.power = power

    def add_obstacles(self, bumps=[]):
        """
        Adds bumps and pothole to the env

        # Args
            bumps, list of (pos, wide, height) 
                if height positive : bump
                          negative : pothole
        """
        bumps.sort()
        self.bumps = bumps

    def set_min_pos(self, min_position= -1.2):
        self.min_position = min_position

    def set_reset_param(self, low=-0.6, high=-0.4):
        self.low_reset = low
        self.high_reset = high

   
    def _reward(self, position, velocity, action):
        modes = self.reward_modes
        reward = 0
        for (mode, prop) in modes:
            if mode == "time":
                reward += prop * - 1.0
            elif mode == "energy":
                reward += prop * - action**2
            elif mode == "distance":
                reward += prop * (position - self.goal_position)
            elif mode == "center":
                reward += prop * (position - (-0.52))**2
            elif mode == "height":
                reward += prop * self._height(position)
            elif mode == "still":
                reward += prop * - (velocity/max_speed)**2
            else:
                print "Unknown mode ".format(mode)
                raise

        return reward

   
    def _obstacle(self, xs):
        """
        Given position, return height if on a bump, else 0
        xs is a numpy array
        """
        def f(x):
            for (pos, wide, height) in self.bumps:
                if np.abs(x - 3*pos) < wide:
                    a = np.pi / (2*wide)
                    return height * np.cos(a * (x - 3*pos))
            return 0

        # DO NOT FORGET TO SPECIFY otype as float64, else convert into int64!!
        g = np.vectorize(f, otypes=[np.float64])
        return g(xs)

    def _obstacle_prime(self, xs):
        """
        Given position, return derivative of the obstacle height
        """
        def f(x):
            for (pos, wide, height) in self.bumps:
                a = np.pi / (2*wide)
                if np.abs(x - 3*pos) < wide:
                    return - height * a * np.sin(a * (x - 3*pos))
            return 0

        g = np.vectorize(f, otypes=[np.float64])
        return g(xs)

    def _height(self, xs):
        """
        compute height as a function of xs + obstacles
        """
        return (np.sin(3*xs) + self._obstacle(3*xs))*.45 + .55 

    def _height_prime(self, xs):
        """
        returns the derivative of the _height to a factor 1/3
        """
        return math.cos(3*xs) + self._obstacle_prime(3*xs)

    def slowdown(self, position, velocity):
        """
        Get slowdown, is the derivative of the height
        """
        return (self._height_prime(position)) * self.slope

    def acceleration(self, action):
        return (action - self.neutral) * self.power

    # -----------------------------------

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        # ------------ MODIFIED ------------
        velocity += self.acceleration(action) - self.slowdown(position, velocity)
        # -------------------------------
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        # ------------ MODIFIED ------------
        reward = self._reward(position, velocity, action)
        # -------------------------------

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=self.low_reset, high=self.high_reset), 0])
        return np.array(self.state)

    
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

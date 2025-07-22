import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class MotorEnvDreamer(gym.Env):
    def __init__(self, render=False):
        self.render = render
        self.dt = 0.001
        self.max_torque = 2.0
        self.max_speed = 10.0
        self.max_angle = np.pi

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-self.max_angle, -self.max_speed, -self.max_angle]),
            high=np.array([self.max_angle, self.max_speed, self.max_angle]),
            dtype=np.float32
        )

        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.arm_id = None
        self.reset()

    def reset(self, seed=None, options=None):
        if self.arm_id is not None:
            p.removeBody(self.arm_id)
        p.loadURDF("plane.urdf")
        self.arm_id = p.loadURDF("pendulum.urdf", useFixedBase=True)


        p.setJointMotorControl2(self.arm_id, 0, p.VELOCITY_CONTROL, force=0)

        theta_init = np.random.uniform(-0.1, 0.1)
        p.resetJointState(self.arm_id, 0, theta_init, 0.0)
        self.theta_target = np.random.uniform(-1.0, 1.0)

        return self._get_obs(), {}

    def _get_obs(self):
        theta, theta_dot = p.getJointState(self.arm_id, 0)[0:2]
        return np.array([theta, theta_dot, self.theta_target], dtype=np.float32)

    def step(self, action):
        torque = float(np.clip(action[0], -1, 1) * self.max_torque)
        p.setJointMotorControl2(self.arm_id, 0, p.TORQUE_CONTROL, force=torque)
        p.stepSimulation()

        obs = self._get_obs()
        theta, theta_dot, theta_target = obs

        reward = -0.1 * ((theta - theta_target)**2 + 0.01 * theta_dot**2 + 0.001 * torque**2)

        done = False
        return obs, reward, done, False, {}

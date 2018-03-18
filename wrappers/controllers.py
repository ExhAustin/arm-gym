import os

import numpy as np
from pyquaternion import Quaternion

from mujoco_py import load_model_from_path, MjSim
from mujoco_py import finctions as mj_functions

from arm_gym.wrappers.controllers.utils import ArmImpController, GripperPDController

class ControllerWrapper():
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()


class SawyerImpControllerWrapper(ControllerWrapper):
    def __init__(self, sawyer_env, arm_name):
        super().__init__(sawyer_env)

        # Sawyer parameters
        self.n_joints = 7
        self.gripper_max_dist = 0.04
        self.n_touch = 20
        self.theta_neutral = np.array([
            0.00221484375, -1.1789130859375, -0.003966796875, 2.1766865234375,
            -0.0081728515625, 0.57493359375, 3.308595703125
            ])
        self.end_effector_name = "effector_origin"

        # Sawyer high-level controller parameters
        self.dt = 0.005
        self.t_epsilon = 0.01 * self.dt

        # Dynamic model
        curr_dir = os.path.abspath(".") + "/"
        arm_file = curr_dir + "../envs/assets/arms/"\
                + arm_name + ".xml"

        self.mj_dynmodel = load_model_from_path(arm_file)
        self.dynsim = MjSim(self.mj_dynmodel)

        # Controller modules
        self.arm_controller = ArmImpController(
                dt=self.dt,
                n_joints=self.n_joints
                end_effector_name = self.end_effector_name
                theta_neutral=self.theta_neutral)
        self.gripper_controller = GripperPDController(
                dt=self.dt,
                n_joints=self.n_joints,
                max_dist=self.gripper_max_dist)

        # Initial state in neutral position
        theta0 = self.theta_neutral
        self.state0 = self.sim.get_state()
        self.state0.qpos[0:7] = theta0

        # Initialize rest by reset
        self.reset()

    def reset(self, state):
        # Reset env
        self.env.reset()
        self.load_state(self.state0)

        # Reset env states
        self.t = 0
        self.is_t0 = True

        # Reset controller modules
        self.arm_controller.reset()
        self.gripper_controller.reset()

        # Take a step to get derivative states
        return self.step()

    def step(self, action=None):
        """
        Args: 
            action - [p_d, r_d, pg_d] (ndarray)
        Returns: 
            new observation - [p, r, sensordata] (ndarray)
        """
        # Extract desired states from input action
        if action is None:
            p_d, r_d = self._forward_kinematics(self.end_effector_name)
            pg_d = self.gripper_max_dist
        else:
            p_d = action[0:3]
            r_d = Quaternion(array=action[3:7])
            pg_d = action[7]

        # Compute control signals
        tau_joints = self.arm_controller.step(self.dynsim, p_d, r_d)
        f_gripper = self.gripper_controller.step(self.dynsim, pg_d)
        ctrl_vec = np.concatenate([tau_joints, f_gripper])

        # Advance simulation
        sensordata = self._simulation_step(ctrl_vec)

        # Extract observation
        p, r = self._forward_kinematics(self.end_effector_name)

        return np.concatenate([p, r.elements, sensordata])

    def save_xml(self):
        pass

    def load_xml(self):
        pass

    def save_state(self):
        return self.env.sim.get_state()

    def load_state(self, state):
        """
        Advance simulation for 1 controller timestep
            (Accesses time through env.sim)
        """
        self.env.sim.set_state(state)
        mj_functions.mj_fwdPosition(self.env.sim.model, self.env.sim.data)

        self._sync_dynmodel()

    def _simulation_step(self, ctrl):
        """
        Advances simulation for 1 controller timestep
            (Accesses time & states through env.sim)

        Args:
            ctrl - vector of control signals to each actuator (ndarray)
        Returns:
            observation - vector of sensor signals from each sensor (ndarray)
        """
        # Advance simulation
        self.t += self.dt
        while(abs(self.env.sim.data.time - self.t) > self.t_epsilon):
            observation = self.env.step(ctrl)

        # Sync dynamic model
        self.dynsim.set_state(self.env.sim.get_state())

        return observation

    def _forward_kinematics(self, body_name):
        """
        Uses dynamic model to calculate forward kinematics

        Args:
            body_name - name of body in MuJoCo model
        Returns:
            p - position of body in world frame (ndarray)
            r - orientation of body in world frame (Quaternion)
        """
        p = self.dynsim.data.get_body_xpos(body_name)
        r = Quaternion(matrix=self.dynsim.data.get_body_xmat(body_name))

        return p, r

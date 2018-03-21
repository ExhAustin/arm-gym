import os
import shutil

import numpy as np
from pyquaternion import Quaternion

from mujoco_py import load_model_from_path, MjSim
from mujoco_py import functions as mj_functions

from arm_gym.wrappers.utils import ArmImpController, GripperPDController
from arm_gym.utils import rotations

class Controller():
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()


class SawyerImpController(Controller):
    def __init__(self, sawyer_env, arm_name):
        super().__init__(sawyer_env)

        # Sawyer parameters
        self.n_joints = 7
        self.pg_max = 0.02
        self.n_touch = 20
        self.theta_neutral = np.array([
            0.00221484375, -1.1789130859375, -0.003966796875, 2.1766865234375,
            -0.0081728515625, 0.57493359375, 3.308595703125
            ])
        self.end_effector_name = "gripper_origin"

        # Sawyer high-level controller parameters
        self.dt = 0.005
        self.t_epsilon = 0.01 * self.dt

        # Dynamic model
        curr_dir = os.path.dirname(__file__) + "/"
        arm_file = curr_dir + "../envs/xmls/arms/"\
                + arm_name + ".xml"

        self.mj_dynmodel = load_model_from_path(arm_file)
        self.dynsim = MjSim(self.mj_dynmodel)

        # Controller modules
        self.arm_controller = ArmImpController(
                dt=self.dt,
                n_joints=self.n_joints,
                end_effector_name = self.end_effector_name,
                theta_neutral=self.theta_neutral)
        self.gripper_controller = GripperPDController(
                dt=self.dt,
                n_joints=self.n_joints,
                p_max=self.pg_max)

        # Initial state in neutral position
        theta0 = self.theta_neutral
        self.state0 = self.env.sim.get_state()
        self.state0.qpos[0:7] = theta0

        # Initialize rest by reset
        self.reset()

    def reset(self):
        # Reset env
        self.env.reset()
        self.load_state(self.state0)

        # Reset env states
        self.t = 0

        # Reset controller modules
        self.arm_controller.reset()
        self.gripper_controller.reset()

        # Reset rendering
        self.rendering = False

        # Take a step to get derivative states
        self.state = self._simulation_step()
        return self.state

    def step(self, action=None):
        """
        Args: 
            action - [p_d, r_d, pg_d] (ndarray)
        Returns: 
            new observation - [p, r, sensordata] (ndarray)
        """
        # Extract current states
        p = self.state[0:3]
        r = Quaternion(array=self.state[3:7])
        pg = self.state[7+self.n_joints]

        # Extract desired states from input action
        if action is None:
            p_d = p
            r_d = r
            pg_d = self.pg_max
        else:
            p_d = action[0:3]
            r_d = Quaternion(array=action[3:7])
            pg_d = action[7]

        # Compute control signals
        tau_joints = self.arm_controller.step(self.dynsim, p, r, p_d, r_d)
        f_gripper = self.gripper_controller.step(pg, pg_d)
        ctrl_vec = np.concatenate([tau_joints, np.array([f_gripper])])

        # Advance simulation
        self.state = self._simulation_step(ctrl_vec)
        return self.state

    def render(self):
        """
        Continuous rendering by using this function to enable render
        """
        self.rendering = True
        self.env.render()

    def save_xml(self, filename):
        """
        Saves model of env to xml file
            (Accesses model_file through env.sim)
        """
        if ".xml" not in filename:
            filename = filename + ".xml"

        shutil.copyfile(self.env.model_file, filename)

    def save_state(self):
        """
        Saves env state
            (Accesses get_state through env.sim)
        """
        return self.env.sim.get_state()

    def load_state(self, state):
        """
        Loads env state
            (Accesses set_state through env.sim)
        """
        self.env.sim.set_state(state)
        mj_functions.mj_fwdPosition(self.env.sim.model, self.env.sim.data)
        self._sync_dynmodel()

    def _get_observation(self, observation):
        """
        Extracts state vector from the environment observation
        """

    def _simulation_step(self, ctrl=None):
        """
        Advances simulation for 1 controller timestep
            (Accesses time & states through env.sim)

        Args:
            ctrl - vector of control signals to each actuator (ndarray)
        Returns:
            new observation - [p, r, sensordata] (ndarray)
                (sensordata - vector of sensor signals from each sensor)
        """
        # Default zero torque control
        if ctrl is None:
            ctrl = np.zeros(self.n_joints+1)

        # Advance simulation
        self.t += self.dt
        while(abs(self.env.sim.data.time - self.t) > self.t_epsilon):
            env_observation = self.env.step(ctrl)
            if self.rendering:
                self.env.render()

        self.rendering = False

        # Sync dynamic model
        self._sync_dynmodel()

        # Return parsed observation
        p, r = self._forward_kinematics(self.end_effector_name)
        return np.concatenate([p, r.elements, env_observation])

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

    def _sync_dynmodel(self):
        self.dynsim.data.qpos[0:7] = self.env.sim.data.qpos[0:7].copy()
        self.dynsim.data.qvel[0:7] = self.env.sim.data.qvel[0:7].copy()
        self.dynsim.data.qacc[0:7] = self.env.sim.data.qacc[0:7].copy()
        mj_functions.mj_fwdPosition(self.dynsim.model, self.dynsim.data)


import numpy as np

#from mujoco_py import load_model_from_path, MjSim, MjViewer
from dm_control import mujoco

class ArmTaskEnv:
    """
    States:
        sensors
    Actions:
        actuators
    """
    def __init__(self, model_file):
        # Load model to MuJoCo
        self.model_file = model_file
        #self.mj_model = load_model_from_path(model_file)
        #self.sim = MjSim(self.mj_model)
        self.sim = mujoco.Physics.from_xml_path(model_file)

        # Parse state and action spaces
        self.state_dim = len(self.sim.data.sensordata)
        self.action_dim = len(self.sim.data.ctrl)

        # Rendering
        self.viewer = None

    def reset(self):
        self.sim.reset()

    def step(self, action):
        # Assign action to actuators
        self.sim.data.ctrl[:] = action.copy()

        # Simulate step
        self.sim.step()

        # Get next state
        next_state = self.sim.data.sensordata.copy()

        return next_state

    def render(self):
        #if self.viewer is None:
        #    self.viewer = MjViewer(self.sim)
        #self.viewer.render()
        pass

    @property
    def time(self):
        return self.sim.get_state().time

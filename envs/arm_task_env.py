import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer

class ArmTaskEnv:
    """
    States:
        sensors
    Actions:
        actuators
    """
    def __init__(self, model_file):
        # Load model to MuJoCo
        self.mj_model = load_model_from_path(model_file)
        self.sim = MjSim(self.model)

        # Rendering
        self.viewer = None

    def reset(self):
        self.sim.reset()

    def step(self, action):
        # Assign action to actuators
        self.sim.data.ctrl = action.copy()

        # Simulate step
        self.sim.step()

        # Get next state
        next_state = self.sim.data.sensordata.copy()

        return next_state

    def render(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    @property
    def time(self):
        return self.sim.get_state().time

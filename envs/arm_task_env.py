import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import functions as mj_functions

class ArmTaskEnv:
    def __init__(self, model_file):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def save_xml(self, filename):
        pass

    def load_xml(self, filename):
        pass

    def save_state(self):
        pass

    def load_state(self, state):
        pass

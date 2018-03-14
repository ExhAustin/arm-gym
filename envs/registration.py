import os
from mujoco_py import load_model_from_path, MjSim, MjViewer

from arm_gym.utils.xml_tools import merge_xmls
from mujoco.arm_task import ArmTaskEnv

def make(arm_name, task_name, options=None):
    curr_dir = os.path.abspath(".") + "/"

    # Load xml files
    arm_file = curr_dir + "mujoco/assets/arms/" + arm_name + ".xml"
    task_file = curr_dir + "mujoco/assets/taskenvs/" + task_name + ".xml"

    # Merge xml files
    out_file = curr_dir + "tmp/" + "merged_tmp.xml"
    merge_xmls(arm_file, task_file, out_file)

    # Create arm environment from xml file
    env = ArmTaskEnv(out_file, options)

    return env

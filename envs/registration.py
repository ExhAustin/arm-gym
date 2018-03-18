import os

import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer

from arm_gym.utils.xml_tools import file2list, shift_xml, merge_xmls
from arm_task_env import ArmTaskEnv

def make(arm_name, task_name, shift=None):
    # Define and locate files directories
    curr_dir = os.path.abspath(".") + "/"
    arm_file = curr_dir + "assets/arms/" + arm_name + ".xml"
    task_file = curr_dir + "assets/taskenvs/" + task_name + ".xml"
    out_file = curr_dir + "tmp/" + "merged_tmp.xml"

    # Load xml files
    arm_lines = file2list(arm_file)
    task_lines = file2list(task_file)

    # Apply shift
    if shift is not None:
        if 'p' in shift:
            p = shift['p']
        else:
            p = np.zeros(3)
        if 'r' in shift:
            r = shift['r']
        else:
            r = np.array([1., 0., 0., 0.])

        task_lines = shift_xml(task_lines, p=p, r=r)

    # Merge xml files and output
    armtask_lines = merge_xmls(arm_lines, task_lines)
    list2file(armtask_lines, out_file)

    # Create arm environment from xml file
    env = ArmTaskEnv(out_file, arm_file)

    return env

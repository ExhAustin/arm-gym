import os

import numpy as np

from arm_gym.utils.xml_tools import file2list, list2file, shift_body, merge_xmls
from arm_gym.envs.arm_task_env import ArmTaskEnv

def make(arm_name=None, task_name=None, filename="merged_tmp", 
        p_shift=None, r_shift=None,
        xml=None):
    if xml is not None:
        print("Creating env object from xml...")
        env = ArmTaskEnv(xml)
        print("Environment created.")
        return env

    print("Generating XML model file...")
    print("\tArm: {}".format(arm_name))
    print("\tTask: {}".format(task_name))
    print("\tTask env shift: {}, {}".format(p_shift, r_shift))

    # Define and locate files directories
    curr_dir = os.path.dirname(__file__) + "/"
    arm_file = curr_dir + "xmls/arms/" + arm_name + ".xml"
    task_file = curr_dir + "xmls/taskenvs/" + task_name + ".xml"
    out_file = curr_dir + "xmls/generated/" + filename + ".xml"

    # Load xml files
    arm_lines = file2list(arm_file)
    task_lines = file2list(task_file)

    # Reallocate mesh directories
    new_mesh_dir = "'../arms/'"
    for i in range(len(arm_lines)):
        if "compiler" in arm_lines[i]:
            tmp_str = arm_lines[i].strip("/>")
            arm_lines[i] = tmp_str + " meshdir=" + new_mesh_dir + "/>"

    # Apply shift
    if p_shift is not None:
        dp = p_shift
    else:
        dp = np.zeros(3)

    if r_shift is not None:
        dr = r_shift
    else:
        dr = np.array([1., 0., 0., 0.])

    task_lines = shift_body(task_lines, p=dp, r=dr)

    # Merge xml files and output
    armtask_lines = merge_xmls(arm_lines, task_lines)
    list2file(armtask_lines, out_file)

    print("XML model file created:".format(out_file))

    # Create arm environment from xml file
    print("Creating env object...")
    env = ArmTaskEnv(out_file)
    print("Environment created.")

    return env

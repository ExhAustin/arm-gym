def file2list(filename):
    f = open(filename, 'r')
    f_ls = []
    for line in f:
        f_ls.append(line)

    return f_ls

def merge_xmls(arm_file, task_file, outfile):
    # Load file contents
    arm_fls = file2list(arm_file)
    task_fls = file2list(task_file)

    # Use arm hyperparameters

    # Merge sections
    section_ls = ["worldbody", "asset", "default", "equality", "actuator", "sensor"]


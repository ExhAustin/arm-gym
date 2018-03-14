import numpy as np

def file2list(filename):
    f = open(filename, 'r')
    line_ls = []
    for line in f:
        line_ls.append(line.strip('\n'))

    return line_ls

def list2file(line_ls, filename):
    f = open(filename, 'w')
    line_count = 0

    for line in line_ls:
        f.write(str(line) + '\n')
        line_count += 1

    return line_count

def shift_xml(line_ls, p=np.zeros(3), r=np.array([1.,0.,0.,0.])):
    """
    Shifts the frame of worldbody in xml file
    """
    # Find worldbody
    for i in range(len(line_ls)):
        if "worldbody" in line_ls[i]:
            start = i
            break

    for j in range(start+1, len(line_ls)):
        if "/worldbody" in line_ls[i]:
            end = j
            break

    # Create shifted body
    pos_str = str(p[0]) + " " + str(p[1]) + " " + str(p[2])
    rot_str = str(r[0]) + " " + str(r[1]) + " " + str(r[2]) + " " + str(r[3])
    header = "\t\t<body pos='{}' quat='{}'>".format(pos_str, rot_str)
    footer = "\t\t</body>"

    # Add new body and indent middle section
    prev_sect = line_ls[:start+1]
    mid_sect = line_ls[start+1:end]
    post_sect = line_ls[end:]

    for i in range(len(mid_sect)):
        mid_sect[i] = "\t" + mid_sect[i]

    merged_ls = prev_sect + header + mid_sect + footer + post_sect

    return merged_ls 

def merge_xmls(file1_lines, file2_lines):
    """ 
    Merge two XML files
    """
    # Remove redundant parts of file2 TODO
    redundant_sections = ['compiler', 'option', 'size']
    i = 0
    while len(task_lines) > i:
        for s in redundant_sections:
            if s in task_lines[i]:
                while ("/" + s) not in line:
                    task_lines.pop(i)
        i += 1

    # Add sections of file2 to file1
    file1_footer = file1_lines.pop(-1)
    file2_lines.pop(0)
    file2_lines.pop(-1)

    return file1_lines + file2_lines + file1_footer


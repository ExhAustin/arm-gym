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

def shift_body(line_ls, body_name="shift", p=np.zeros(3), r=np.array([1.,0.,0.,0.])):
    """
    Shifts the frame of a body in an xml file
    """
    # Generate strings
    pos_str = "{} {} {}".format(str(p[0]), str(p[1]), str(p[2]))
    rot_str = "{} {} {} {}".format(str(r[0]), str(r[1]), str(r[2]), str(r[3]))

    # Find body and edit frame
    for i in range(len(line_ls)):
        if (body_name in line_ls[i]) and ("<body" in line_ls[i]):
            tmp_str = line_ls[i].strip(">")
            line_ls[i] = tmp_str + " pos='{}' quat='{}'>".format(pos_str, rot_str)

    return line_ls

    """
    # Find worldbody
    for i in range(len(line_ls)):
        if "worldbody" in line_ls[i]:
            start = i
            break

    for j in range(start+1, len(line_ls)):
        if "/worldbody" in line_ls[j]:
            end = j
            break

    # Create shifted body
    pos_str = str(p[0]) + " " + str(p[1]) + " " + str(p[2])
    rot_str = str(r[0]) + " " + str(r[1]) + " " + str(r[2]) + " " + str(r[3])
    header = "\t\t<body name='shift_frame' pos='{}' quat='{}'>".format(pos_str, rot_str)
    footer = "\t\t</body>"

    # Add new body and indent middle section
    prev_sect = line_ls[:start+1]
    mid_sect = line_ls[start+1:end]
    post_sect = line_ls[end:]

    for i in range(len(mid_sect)):
        mid_sect[i] = "\t" + mid_sect[i]

    merged_ls = prev_sect + [header] + mid_sect + [footer] + post_sect

    return merged_ls 
    """

def merge_xmls(file1_lines, file2_lines):
    """ 
    Merge two XML files
    """
    # Remove redundant parts of file2
    redundant_sections = ['<compiler', '<option', '<size']
    i = 0
    while True:
        for s in redundant_sections:
            if s in file2_lines[i]:
                if "/>" in file2_lines[i]:
                    file2_lines.pop(i)
                else:
                    while ("/" + s) not in file2_lines[i]:
                        file2_lines.pop(i)
        i += 1
        if len(file2_lines) <= i:
            break

    # Add sections of file2 to file1
    file1_footer = [file1_lines.pop(-1)]
    file2_lines.pop(0)
    file2_lines.pop(-1)

    return file1_lines + file2_lines + file1_footer


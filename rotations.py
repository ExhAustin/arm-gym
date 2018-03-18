import numpy as np
import math
from pyquaternion import Quaternion

# Converts quaternions to rotational velocities
def quat2vec(q):
    return q.angle * q.axis

# Converts rotational velocites to quaternions
def vec2quat(v):
   v_abs = np.linalg.norm(v) 
   return Quaternion(axis=v, angle=v_abs)

# Normalize angle differences to [-pi, pi]
def angdiff(theta):
    return np.mod(theta+np.pi, 2*np.pi) - np.pi

# Get difference between two quaternions as rotational velocity vector
def quatdiff(q1, q2):
    return quat2vec(q1 * q2.inverse)

# Converts quaternion to euler angles (from Wikipedia)
def quat2euler(quat):
    if type(quat).__module__ == np.__name__:
        quat = Quaternion(array=quat)

    w = quat.elements[0]
    x = quat.elements[1]
    y = quat.elements[2]
    z = quat.elements[3]

    ysqr = y * y
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    Rx = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Ry = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Rz = math.atan2(t3, t4)
    
    return np.mod(np.array([Rx, Ry, Rz]), 2*np.pi)

# Converts euler angles to quaternion (from Wikipedia)
def euler2quat(r):
    roll = r[0]
    pitch = r[1]
    yaw = r[2]

    cy = math.cos(yaw * 0.5);
    sy = math.sin(yaw * 0.5);
    cr = math.cos(roll * 0.5);
    sr = math.sin(roll * 0.5);
    cp = math.cos(pitch * 0.5);
    sp = math.sin(pitch * 0.5);

    w = cy * cr * cp + sy * sr * sp;
    x = cy * sr * cp - sy * cr * sp;
    y = cy * cr * sp + sy * sr * cp;
    z = sy * cr * cp - cy * sr * sp;

    return Quaternion(w,x,y,z); 

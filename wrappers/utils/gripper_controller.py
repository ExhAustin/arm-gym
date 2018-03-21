import numpy as np

# Gripper controller
class GripperPDController:
    def __init__(self, dt, n_joints, p_max):
        self.dt = dt
        self.n_joints = n_joints
        self.p_max = p_max

        # Controller gains
        self.Kp = 400
        self.Kd = 5

        # Initialize memory by reset
        self.reset()

    def reset(self):
        self.prev_pg_e = None

    def step(self, pg, pg_d):
        # Get error
        pg_d = np.clip(pg_d, 0, self.p_max)
        pg_e = pg_d - pg

        # Derivative
        if self.prev_pg_e is None:
            dpg_e = 0
        else:
            dpg_e = (pg_e - self.prev_pg_e)/self.dt

        # Update memory
        self.prev_pg_e = pg_e

        # PD control
        return self.Kp*pg_e + self.Kd*dpg_e
        #return self.Kp*pg_e + self.Kd*dpg_e - f_d

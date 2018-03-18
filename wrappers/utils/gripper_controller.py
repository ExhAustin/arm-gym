import numpy as np

# Gripper controller
class GripperController:
    def __init__(self, dt, n_joints, max_dist):
        self.dt = dt
        self.n_joints = n_joints
        self.max_pg = max_dist/2.0

        # Controller gains
        self.Kp = 400
        self.Kd = 2
        #self.Kp = 25
        #self.Kd = 0.1

        # Initialize memory by reset
        self.reset()

    def reset(self):
        self.prev_pg_e = None

    def step(self, dynsim, pg_d):
        # Get error
        pg_d = np.clip(pg_d, 0, self.max_pg)
        pg = dynsim.get_state().qpos[self.n_joints]
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

import numpy as np
from mujoco_py import functions as mj_functions

from arm_sim.utils import rotations

class ArmImpController:
    def __init__(self, dt, n_joints, end_effector_name, theta_neutral=None):
        self.dt = dt
        self.n_joints = n_joints
        self.end_effector_name = end_effector_name
        if theta_neutral is None:
            self.theta_neutral = np.zeros(n_joints)
        else:
            self.theta_neutral = theta_neutral

        # Position controller gain matrices
        self.Kp_theta = 1000*np.eye(self.n_joints)
        self.Kd_theta = 350*np.eye(self.n_joints)
        #self.Kp_x = 10*np.diag([1,1,1,10,10,10])
        #self.Kd_x = 3.5*np.diag([1,1,1,10,10,10])

        # Limits & decays
        self.p_e_max = 0.05
        self.r_e_max = 0.3

        # Initialize by reset
        self.reset()

    def reset(self):
        """
        Reset controller states
        """
        self.p_d_prev = None
        self.r_d_prev = None
        self.dx_d_prev = np.zeros(6)
        self.prev_theta_e = None
        self.if_e = np.zeros(6)

    def step(self, dynsim, p_d, r_d):
        """
        Controller step
        """
        # Warmstart
        if self.p_d_prev is None:
            self.p_d_prev = p_d
        if self.r_d_prev is None:
            self.r_d_prev = r_d

        # Compute position error
        p, r = self._forward_kinematics(dynsim, self.end_effector_name)

        p_e = p_d - p
        r_e = rotations.quatdiff(r_d, r)

        x_e = np.concatenate([p_e, r_e])

        # Compute desired acceleration
        dp_d = (p_d - self.p_d_prev) / self.dt
        dr_d = rotations.quatdiff(r_d, self.r_d_prev) / self.dt
        dx_d = np.concatenate([dp_d, dr_d])
        ddx_d = (dx_d - self.dx_d_prev) / self.dt

        # Let fancy low-level controller handle the rest
        tau_output = self.position_controller(dynsim, x_e, ddx_d)

        # Update memory
        self.p_d_prev = p_d
        self.r_d_prev = r_d
        self.dx_d_prev = dx_d

        return tau_output

    def _position_controller(self, dynsim, x_e, ddx_d):
        """
        Position controller
        """
        # Clip errors to ensure stability
        x_e[0:3] = vec_softclip(x_e[0:3], self.p_e_max)
        x_e[3:6] = vec_softclip(x_e[3:6], self.r_e_max)

        # Compute Jacobian
        J = self._get_jacobian(dynsim, self.end_effector_name)

        # Compute pose error (including null space error)
        theta_e_end = np.linalg.pinv(J) @ x_e
        theta_e_null = self._null_error(dynsim, J)
        theta_e = theta_e_end + theta_e_null

        # Get position error derivative
        if self.prev_theta_e is None:
            self.prev_theta_e = theta_e
        dtheta_e = (theta_e - self.prev_theta_e)/self.dt

        # PD acceleration term
        #self.Kp_theta = env.J.T @ self.Kp_x @ env.J
        #self.Kd_theta = env.J.T @ self.Kd_x @ env.J
        ddtheta_pd = self.Kp_theta @ theta_e + self.Kd_theta @ dtheta_e

        # Inverse dynamics
        ddtheta_d = np.linalg.pinv(env.J) @ ddx_d
        tau_p = self._inv_dynamics(dynsim, ddtheta_d + ddtheta_pd)

        # Update memory
        self.prev_theta_e = theta_e

        return tau_p

    def _null_error(self, dynsim, J):
        """
        Null space motion feedforward term
        """
        # Compute error from current pose to neutral pose
        theta = dynsim.get_state().qpos[0:self.n_joints]
        theta_e = rotations.angdiff(self.theta_neutral - theta)

        # Project error to null space
        null_projection = np.eye(7) - (np.linalg.pinv(J) @ J)
        theta_null_e = null_projection @ theta_e

        return theta_null_e

    def _inv_dynamics(self, dynsim, ddtheta_d):
        """
        Inverse dynamics feedforward term
        """
        # Set desired acceleration
        dynsim.data.qacc[0:self.n_joints] = ddtheta_d

        # Compute inverse dynamics
        mj_functions.mj_inverse(dynsim.model, dynsim.data)
        qfrc_inv = dynsim.data.qfrc_inverse[0:self.n_joints]
        tau_invdyn = qfrc_inv

        """
        qfrc_pass = env.dynsim.data.qfrc_passive[0:self.n_joints]
        mj_functions.mj_passive(dynsim.model, dynsim.data)
        tau_invdyn = qfrc_inv + qfrc_pass
        tau_invdyn = qfrc_inv - qfrc_pass
        """

        return tau_invdyn

    def _forward_kinematics(self, dynsim, body_name):
        """
        Uses dynamic model to calculate forward kinematics

        Args:
            body_name - name of body in MuJoCo model
        Returns:
            p - position of body in world frame (ndarray)
            r - orientation of body in world frame (Quaternion)
        """
        p = dynsim.data.get_body_xpos(body_name)
        r = Quaternion(matrix=dynsim.data.get_body_xmat(body_name))

        return p, r

    def _get_jacobian(self, dynsim, body_name):
        J = np.zeros([6, self.n_joints])

        Jp = dynsim.data.get_body_jacp(body_name)
        Jr = dynsim.data.get_body_jacp(body_name)

        J[0:3,:] = Jp.reshape([3,-1])[:,0:self.n_joints]
        J[3:6,:] = Jr.reshape([3,-1])[:,0:self.n_joints]

        return J

def vec_softclip(a, a_max, leak=0.1):
    """
    Clip length of vector with a soft constraint (inverted ELU)
    """
    a_len = np.linalg.norm(a)

    if a_len > a_max:
        max_leak = a_max * leak
        clipped_len = a_max +\
                max_leak * ( 1 - np.exp(-(1/max_leak)*(a_max-a_len)) )
        return a * (clipped_len / a_len)
    else:
        return a

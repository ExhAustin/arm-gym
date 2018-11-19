import numpy as np
from pyquaternion import Quaternion

from mujoco_py import functions as mjlib
#from dm_control.mujoco.wrapper.mjbindings import mjlib

from arm_gym.utils import rotations

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
        self.Kp_theta = 500*np.eye(self.n_joints)
        self.Kd_theta = 100*np.eye(self.n_joints)
        self.Kp_x = 1000*np.diag([1,1,1,1,1,1])
        self.Kd_x = 50*np.diag([1,1,1,1,1,1])

        # Limits & decays
        self.fp_max = 30
        self.fr_max = 0.1*self.fp_max

        # Initialize by reset
        self.reset()

    def reset(self):
        """ Reset controller states """
        self.p_d_prev = None
        self.r_d_prev = None
        self.dx_d_prev = np.zeros(6)

        self.theta_e_prev = None

        self.x_e_prev = None
        self.J_prev = None
        self.theta_e_end_prev = None
        self.theta_e_null_prev = None


    def step(self, dynsim, p, r, p_d, r_d):
        """ Controller step """
        # Warmstart
        if self.p_d_prev is None:
            self.p_d_prev = p_d
        if self.r_d_prev is None:
            self.r_d_prev = r_d

        # Compute position error
        p_e = p_d - p
        r_e = rotations.quatdiff(r_d, r)

        x_e = np.concatenate([p_e, r_e])

        # Compute desired acceleration
        dp_d = (p_d - self.p_d_prev) / self.dt
        dr_d = rotations.quatdiff(r_d, self.r_d_prev) / self.dt
        dx_d = np.concatenate([dp_d, dr_d])
        ddx_d = (dx_d - self.dx_d_prev) / self.dt

        # Let fancy low-level controller handle the rest
        tau_output = self._position_controller(dynsim, x_e, ddx_d)

        # Update memory
        self.p_d_prev = p_d
        self.r_d_prev = r_d
        self.dx_d_prev = dx_d

        return tau_output

    def _position_controller(self, dynsim, x_e, ddx_d):
        """ Position controller """
        #"""
        # Compute Jacobian
        J = self._get_jacobian(dynsim, self.end_effector_name)

        # Compute pose error (including null space error)
        theta_e_end = np.linalg.pinv(J).dot(x_e)
        theta_e_null = self._null_error(dynsim, J)
        theta_e = theta_e_end + theta_e_null

        # Get pose error derivative
        if self.theta_e_prev is None:
            self.theta_e_prev = theta_e
        dtheta_e = (theta_e - self.theta_e_prev)/self.dt

        # PD acceleration term
        ddtheta_pd = self.Kp_theta.dot(theta_e) + self.Kd_theta.dot(dtheta_e)

        # Inverse dynamics
        ddtheta_d = np.linalg.pinv(J).dot(ddx_d)
        tau = self._inv_dynamics(dynsim, ddtheta_d + ddtheta_pd)

        # Update memory
        self.theta_e_prev = theta_e
        """
        # Compute Jacobian
        J = self._get_jacobian(dynsim, self.end_effector_name)

        # State error derivative
        if self.x_e_prev is None:
            self.x_e_prev = x_e
        dx_e = (x_e - self.x_e_prev) / self.dt
        
        # Cartesian impedance
        F_imp = self.Kd_x.dot(dx_e) + self.Kp_x.dot(x_e)
        F_imp[0:3] = vec_softclip(F_imp[0:3], self.fp_max)
        F_imp[3:6] = vec_softclip(F_imp[3:6], self.fr_max)
        tau_imp = J.T.dot(F_imp)

        # Desired null space joint acceleration
        theta_e_null = self._null_error(dynsim, J)
        if self.theta_e_null_prev is None:
            self.theta_e_null_prev = theta_e_null
        dtheta_e_null = (theta_e_null - self.theta_e_null_prev)/self.dt

        ddtheta_null_d = self.Kp_theta.dot(theta_e_null) +\
                self.Kd_theta.dot(dtheta_e_null)

        # Inverse dynamics
        ddtheta_d = ddtheta_null_d + np.linalg.pinv(J).dot(ddx_d)
        tau_invdyn = self._inv_dynamics(dynsim, ddtheta_d)

        # Sum torques
        tau = tau_invdyn + tau_imp

        # Update memory
        self.x_e_prev = x_e
        self.theta_e_null_prev = theta_e_null
        """


        return tau

    def _null_error(self, dynsim, J):
        """ Null space motion feedforward term """
        # Compute error from current pose to neutral pose
        theta = dynsim.get_state().qpos[0:self.n_joints]
        theta_e = rotations.angdiff(self.theta_neutral - theta)

        # Project error to null space
        null_projection = np.eye(7) - (np.linalg.pinv(J).dot(J))
        theta_null_e = null_projection.dot(theta_e)

        return theta_null_e

    def _inv_dynamics(self, dynsim, ddtheta_d):
        """ Inverse dynamics feedforward term """
        # Set desired acceleration
        dynsim.data.qacc[0:self.n_joints] = ddtheta_d

        # Compute inverse dynamics
        mjlib.mj_inverse(dynsim.model, dynsim.data)
        qfrc_inv = dynsim.data.qfrc_inverse[0:self.n_joints]
        tau_invdyn = qfrc_inv

        """
        qfrc_pass = env.dynsim.data.qfrc_passive[0:self.n_joints]
        mjlib.mj_passive(dynsim.model, dynsim.data)
        tau_invdyn = qfrc_inv + qfrc_pass
        tau_invdyn = qfrc_inv - qfrc_pass
        """

        return tau_invdyn

    def _get_jacobian(self, dynsim, body_name):
        """ Acquire Jacobian from joint space to input body pose """
        J = np.zeros([6, self.n_joints])

        Jp = dynsim.data.get_body_jacp(body_name)
        Jr = dynsim.data.get_body_jacr(body_name)

        J[0:3,:] = Jp.reshape([3,-1])[:,0:self.n_joints]
        J[3:6,:] = Jr.reshape([3,-1])[:,0:self.n_joints]

        return J

    def _get_qM(self, dynsim, J):
        """ Joint mass matrix for current pose """
        qM = np.zeros(self.n_joints * self.n_joints + 100)
        mjlib.mj_fullM(dynsim.model, qM, dynsim.data.qM)
        return qM[0:49].reshape(self.n_joints, self.n_joints)

def vec_softclip(a, a_max, leak=0.1):
    """ Clip length of vector with a soft constraint (inverted ELU) """
    a_len = np.linalg.norm(a)

    if a_len > a_max:
        max_leak = a_max * leak
        clipped_len = a_max +\
                max_leak * ( 1 - np.exp((1/max_leak)*(a_max-a_len)) )
        return a * (clipped_len / a_len)
    else:
        return a

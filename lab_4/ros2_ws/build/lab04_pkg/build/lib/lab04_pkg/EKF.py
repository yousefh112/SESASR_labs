import numpy as np
from numpy.linalg import inv
import math
from lab04_pkg.velocity4ekf import velocity_motion_model_wrapper, jacobian_Gt, jacobian_Vt     
from lab04_pkg.landmark_model import landmark_model_jacobian

class RobotEKF:
    """Extended Kalman Filter for robot localization and mapping."""
    
    def __init__(self, initial_mu=None):
        """
        Initialize the EKF with motion and measurement models.
        
        Args:
            initial_mu: Initial state estimate [x, y, theta]. Defaults to zero vector.
        """
        # Motion model and Jacobian functions
        self.eval_gux = velocity_motion_model_wrapper   # 5-state motion model wrapper
        self.eval_Gt = jacobian_Gt                       # Jacobian of motion model w.r.t. state
        self.eval_Vt = jacobian_Vt                       # Jacobian of motion model w.r.t. noise

        # State and control dimensions
        self.dim_x = 3  # State: [x, y, theta]
        self.dim_u = 2  # Control: [velocity, angular_velocity]

        # Initialize state estimate
        if initial_mu is None:
            self.mu = np.zeros(3)
        else:
            self.mu = np.array(initial_mu, dtype=float)
            
        # Initialize covariance matrices
        self.Sigma = np.eye(self.dim_x)  # State covariance
        self.Mt = np.eye(self.dim_u)     # Motion noise covariance
        self._I = np.eye(self.dim_x)     # Identity matrix for efficiency

    def predict(self, u, sigma_u, g_extra_args=()):
        """
        Prediction step: propagate state and covariance through motion model.
        
        Args:
            u: Control input [velocity, angular_velocity]
            sigma_u: Motion noise parameters
            g_extra_args: Additional arguments for motion model
        """
        # Update state estimate using motion model
        self.mu = self.eval_gux(self.mu, u, sigma_u, *g_extra_args)
 
        # Compute Jacobians for covariance propagation
        Gt = self.eval_Gt(self.mu, u, *g_extra_args)    # Jacobian w.r.t. state
        Vt = self.eval_Vt(self.mu, u, *g_extra_args)    # Jacobian w.r.t. noise
        
        # Update covariance: Sigma = Gt*Sigma*Gt^T + Vt*Mt*Vt^T
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(), residual=np.subtract, **kwargs):
        """
        Update step: correct state and covariance using measurement.
        
        Args:
            z: Measurement vector
            eval_hx: Function to compute expected measurement
            eval_Ht: Function to compute measurement Jacobian
            Qt: Measurement noise covariance
            Ht_args: Arguments for Ht evaluation
            hx_args: Arguments for hx evaluation
            residual: Function to compute measurement residual
        """
        # Convert scalar measurements to arrays for consistent processing
        if np.isscalar(z):
            z = np.asarray([z], float)
            Qt = np.atleast_2d(Qt).astype(float)

        # Compute Kalman gain: K = Sigma*Ht^T * (Ht*Sigma*Ht^T + Qt)^-1
        Ht = eval_Ht(*Ht_args)
        SigmaHT = self.Sigma @ Ht.T
        self.S = Ht @ SigmaHT + Qt  # Innovation covariance
        self.K = SigmaHT @ np.linalg.pinv(self.S)  # Kalman gain (using pseudo-inverse for robustness)

        # Compute measurement residual: y = z - z_hat
        z_hat = eval_hx(*hx_args)
        if np.isscalar(z_hat):
            z_hat = np.asarray([z_hat], float)
        y = residual(z, z_hat, **kwargs)
        
        # Update state: mu = mu + K*y
        self.mu = self.mu + self.K @ y

        # Update covariance using Joseph form (numerically stable):
        # Sigma = (I - K*H)*Sigma*(I - K*H)^T + K*Qt*K^T
        I_KH = self._I - self.K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T
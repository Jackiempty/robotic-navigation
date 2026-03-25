# PathTracking/controller_lqr_bicycle.py
import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, model, Q=None, R=None, control_state='steering_angle'):
        self.path = None
        if control_state == 'steering_angle':
            self.Q = np.eye(2)
            self.R = np.eye(1)
            # TODO 4.4.1: Tune LQR Gains
            self.Q[0,0] = 10
            self.Q[1,1] = 1
            self.R[0,0] = 10
        elif control_state == 'steering_angular_velocity':
            self.Q = np.eye(3)
            self.R = np.eye(1)
            # TODO 4.4.4: Tune LQR Gains
            self.Q[0,0] = 15
            self.Q[1,1] = 1
            self.Q[2,2] = 0.1
            self.R[0,0] = 1
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.dt = model.dt
        self.l = model.l
        self.control_state = control_state
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.current_idx = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])
        
        if self.control_state == 'steering_angle':
            # TODO 4.4.1: LQR Control for Bicycle Kinematic Model with steering angle as control input
            self.current_idx = min_idx
            v_ = max(v, 0.1)
            yaw_rad = np.deg2rad(yaw)
            target_yaw_rad = np.deg2rad(target[2])
            A = np.array([
                [1.0, v_ * self.dt],
                [0.0, 1.0]
            ])
            B = np.array([
                [0.0],
                [(v_ * self.dt) / self.l]
            ])
            dx = x - target[0]
            dy = y - target[1]
            e = -np.sin(target_yaw_rad) * dx + np.cos(target_yaw_rad) * dy
            th_e = utils.angle_norm(yaw - target[2])
            th_e_rad = np.deg2rad(th_e)
            X = np.array([
                [e],
                [th_e_rad]
            ])

            # LQR
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
            u = -(K @ X)[0, 0]   # rad

            # ---------- Feedforward ----------
            if min_idx + 1 < len(self.path):
                next_target = self.path[min_idx + 1]
                dx2 = next_target[0] - target[0]
                dy2 = next_target[1] - target[1]
                ds = np.hypot(dx2, dy2)
                dyaw = utils.angle_norm(next_target[2] - target[2])
                dyaw = np.deg2rad(dyaw)
                kappa = dyaw / ds if ds > 1e-4 else 0.0
            else:
                kappa = 0.0

            delta_ff = np.arctan(self.l * kappa)
            delta_rad = u + delta_ff
            next_delta = np.rad2deg(delta_rad)
            # [end] TODO 4.4.1
        elif self.control_state == 'steering_angular_velocity':
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
            self.current_idx = min_idx
            v_ = max(v, 0.1)
            yaw_rad = np.deg2rad(yaw)
            target_yaw_rad = np.deg2rad(target[2])
            delta_rad = np.deg2rad(delta)
            A = np.array([
                [1.0, v_ * self.dt, 0.0],
                [0.0, 1.0, (v_ * self.dt) / self.l],
                [0.0, 0.0, 1.0]
            ])
            B = np.array([
                [0.0],
                [0.0],
                [self.dt]
            ])
            dx = x - target[0]
            dy = y - target[1]
            e = -np.sin(target_yaw_rad) * dx + np.cos(target_yaw_rad) * dy
            th_e = utils.angle_norm(yaw - target[2])
            th_e_rad = np.deg2rad(th_e)
            X = np.array([
                [e],
                [th_e_rad],
                [delta_rad]
            ])
            # LQR
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
            u = -(K @ X)[0, 0]   # rad/s
            next_delta_rad = delta_rad + u * self.dt
            next_delta = np.rad2deg(next_delta_rad)
            # [end] TODO 4.4.4
        
        return next_delta

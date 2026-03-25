# PathTracking/controller_lqr_basic.py
import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBasic(Controller):
    def __init__(self, model, Q=np.eye(2), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.Q[0,0] = 100
        self.Q[1,1] = 5
        self.R = R*2000
        self.pe = 0
        self.pth_e = 0
        self.dt = model.dt
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
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

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]
        
        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx

        target = self.path[min_idx]
        
        # Optional TODO: LQR Control for Basic Kinematic Model
        # You can implement this if you want to use LQR for basic kinematic model in F1 Challenge
        # 1. 防呆與初始化
        v_ = max(v, 0.1)
        
        # 2. 建立基礎模型的離散狀態空間矩陣 A 與 B
        # 注意 B 矩陣不再需要 v/L，因為控制輸入直接就是角速度 w
        A = np.array([
            [1.0, v_ * self.dt],
            [0.0, 1.0]
        ])
        B = np.array([
            [0.0],
            [self.dt]
        ])

        # 3. 計算誤差 X = [e, theta_e]^T (Frenet Frame 投影)
        target_yaw_rad = np.deg2rad(target[2])
        yaw_rad = np.deg2rad(yaw)
        
        dx = x - target[0]
        dy = y - target[1]
        
        e = -np.sin(target_yaw_rad) * dx + np.cos(target_yaw_rad) * dy
        th_e = utils.angle_norm(yaw - target[2])
        th_e_rad = np.deg2rad(th_e)
        
        X = np.array([
            [e],
            [th_e_rad]
        ])

        # 4. 求解 DARE 計算 LQR 增益 K
        P = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        
        # 計算 LQR 輸出的反饋角速度 (rad/s)
        u_rad_s = -(K @ X)[0, 0]

        # 5. 前饋控制 (Feedforward)
        # 計算基礎模型過彎所需的理想角速度 w_ff = v * kappa
        if self.current_idx + 1 < len(self.path):
            next_target = self.path[self.current_idx + 1]
            dx2 = next_target[0] - target[0]
            dy2 = next_target[1] - target[1]
            ds = np.hypot(dx2, dy2)
            
            dyaw = utils.angle_norm(next_target[2] - target[2])
            dyaw_rad = np.deg2rad(dyaw)
            
            kappa = dyaw_rad / ds if ds > 1e-4 else 0.0
        else:
            kappa = 0.0
            
        w_ff_rad_s = v_ * kappa

        # 6. 總輸出角速度 = LQR反饋 + 曲率前饋 (並轉回 deg/s)
        total_w_rad_s = u_rad_s + w_ff_rad_s
        next_w = np.rad2deg(total_w_rad_s)
        return next_w

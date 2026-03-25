# PathTracking/controller_pure_pursuit_basic.py
import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBasic(Controller):
    def __init__(self, model, 
                 # Optional TODO: Tune Pure Pursuit Gain
                 kp=0.5, Lfc=1.0):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

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
        
        Ld = self.kp*v + self.Lfc

        # Optional TODO: Pure Pursuit Control for Basic Kinematic Model
        # You can implement this if you want to use Pure Pursuit for basic kinematic model in F1 Challenge
        target_idx = min_idx
        for i in range(min_idx, len(self.path)):
            dist = np.hypot(self.path[i][0] - x, self.path[i][1] - y)
            if dist >= Ld:
                target_idx = i
                break
        
        target = self.path[target_idx]

        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        alpha_deg = theta_target - yaw
        actual_Ld = np.hypot(target[0] - x, target[1] - y)
        if actual_Ld < 1e-4:
            actual_Ld = 1e-4

        alpha_rad = np.deg2rad(alpha_deg)
        w_rad = (2 * v * np.sin(alpha_rad)) / actual_Ld
        next_w = np.rad2deg(w_rad)
        
        return next_w

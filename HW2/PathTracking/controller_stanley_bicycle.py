# PathTracking/controller_stanley_bicycle.py 
import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Stanley Gain
                 kp=4.0):
        self.path = None
        self.kp = kp
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Front Wheel Target Locally
        front_x = x + self.l*np.cos(np.deg2rad(yaw))
        front_y = y + self.l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta)) if np.cos(np.deg2rad(delta)) != 0 else v
        
        min_idx, min_dist = utils.search_nearest_local(self.path, (front_x,front_y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # TODO 4.3.1: Stanley Control for Bicycle Kinematic Model
        yaw_rad = np.deg2rad(yaw)
        if min_idx + 1 < len(self.path):
            next_target = self.path[min_idx + 1]
        else:
            next_target = target
        theta_p_rad = np.arctan2(next_target[1] - target[1], next_target[0] - target[0])
        theta_e_rad = theta_p_rad - yaw_rad
        theta_e_rad = np.arctan2(np.sin(theta_e_rad), np.cos(theta_e_rad))
        dx = front_x - target[0]
        dy = front_y - target[1]
        e_f = -(np.cos(theta_p_rad) * dy - np.sin(theta_p_rad) * dx)
        epsilon = 1e-6 
        delta_rad = theta_e_rad + np.arctan((self.kp * e_f) / (vf + epsilon))
        next_delta = np.rad2deg(delta_rad)
        # [end] TODO 4.3.1
    
        return next_delta

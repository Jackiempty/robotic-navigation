# Simulation/kinematic_bicycle.py
import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBicycle(KinematicModel):
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.05
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        # TODO 2.3.1: Bicycle Kinematic Model
        v, w, x, y, yaw = state.v, state.w, state.x, state.y, state.yaw
        a, delta = cstate.a, cstate.delta
        x_delta = v * np.cos(np.deg2rad(yaw)) * self.dt
        y_delta = v * np.sin(np.deg2rad(yaw)) * self.dt
        yaw_delta = (v / self.l) * np.tan(np.deg2rad(delta)) * self.dt

        x += x_delta
        y += y_delta
        yaw += np.rad2deg(yaw_delta)
        v += a * self.dt
        w = np.rad2deg((v / self.l) * np.tan(np.deg2rad(delta)))
        # [end] TODO 2.3.1
        state_next = State(x, y, yaw, v, w)
        return state_next

import numpy as np
from quad_msgs.msg import Calf
import casadi as ca
import scipy

eps = 1e-10
sep = "\n-------------------------\n"
sep2 = "-----"


def msg2Mat(mat, dtype=float, traspose=True, diagnostic=False, shape=(-1, 26)):
    validator = mat[-1]

    if traspose:
        res = np.array(mat[:-1], dtype=dtype).reshape(shape).T
    else:
        res = np.array(mat[:-1], dtype=dtype).reshape((shape[1], shape[0]))

    if (diagnostic):
        print(sep)
        print(res)
        print(sep2)
        print(res.shape)
        print(validator)

    if abs(res.sum() + res[0, :].sum() - validator) >= eps:
        raise Exception("message from Calf_observation is corrupted")
    return res


def msg2Vec(vec):
    validator = vec[-1]
    res = np.array(vec[:-1])
    if abs(res.sum() - validator) >= eps:
        raise Exception("message from Calf_observation is corrupted")
    return res


class MPCParams:
    def __init__(self,  robotStateMsg: Calf) -> None:
        self.ref_body_plan = msg2Mat(robotStateMsg.ref_body_plan)
        self.contact_schedule = msg2Mat(
            robotStateMsg.contact_schedule, dtype=int, traspose=False)
        self.grf_positions_world = msg2Mat(
            robotStateMsg.grf_positions_world)


class Observation:
    def __init__(self, robotStateMsg: Calf, mpc_param: MPCParams) -> None:
        if robotStateMsg != None:
            self.index = robotStateMsg.header.seq
            self.state = msg2Vec(
                robotStateMsg.current_full_state)[0:12].reshape((12, 1))
            grf_pos = mpc_param.grf_positions_world.T[:, 0].reshape(12, 1)
            ref_b_pos = mpc_param.ref_body_plan.T[:, 1].reshape(12, 1)
            self.full_state = np.vstack([self.state, grf_pos, ref_b_pos])


class ObjParams():
    def __init__(self, robotStateMsg: Calf) -> None:
        self.index = robotStateMsg.header.seq
        self.Q = msg2Vec(robotStateMsg.Q)
        self.R = msg2Vec(robotStateMsg.R)
        self.x_nom = msg2Vec(robotStateMsg.x_nom).reshape((12, 1))
        self.u_nom = msg2Vec(robotStateMsg.u_nom).reshape((12, 1))
        self.Q = np.diag(self.Q)
        self.R = np.diag(self.R)

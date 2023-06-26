
import numpy as np
import rospy
import casadi as ca
import pickle
from quad_msgs.msg import CalfResponse


class A1system():
    """
    System class: unitree A1

    """

    def __init__(
        self,
        A1_topic_name_res,
        dim_state: int,
        dim_input: int,
        dim_output: int,
    ):
        print("a1 system init")
        rospy.init_node("A1_planner")
        self.name = "unitree A1"
        self.action = None
        self.msg_time = None
        self.pub_action = rospy.Publisher(
            A1_topic_name_res, CalfResponse, queue_size=1)
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        with open("./src/rql_mpc/src/casadi_rhs.txt", "rb") as outf:
            self.casadi_rhs = pickle.load(outf)
        print("a1 system is ready")

    def compute_dynamics(self, time, full_state, action):
        if full_state.shape[0] != 36:
            print(full_state.shape)
            raise Exception("system shape input error")
        state = full_state[:12]
        params = full_state[12:24]

        if (type(action) == ca.MX) or (type(action) == ca.DM):
            parameters = ca.vcat([params, action])
            outtype = type(action)
        elif type(params) == np.ndarray:
            parameters = np.vstack([params, action])
            outtype = np.ndarray
        rhs = self.casadi_rhs(state, parameters)

        if outtype == ca.MX:
            return rhs
        else:
            return rhs.full()

    def out(self, state, time=None, action=None):
        return state

    def publishAction(self):
        grf_plan, body_plan = self.action, self.body_plan
        grf_plan = np.append(grf_plan, sum(grf_plan))
        body_plan = np.append(body_plan, sum(body_plan))
        msg = CalfResponse()
        msg.body_plan = body_plan
        msg.grf_plan = grf_plan
        self.pub_action.publish(msg)

    def receive_action(self, action, time):
        self.action = action
        self.msg_time = time

    def receive_body_plan(self, body_plan):
        self.body_plan = body_plan

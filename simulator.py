import rospy
from quad_msgs.msg import Calf
import numpy as np
import __ros_utils as utils
from collections import deque


class A1simulator():

    def __init__(
            self,
            system,
            dt,
            A1_topic_name_req,
            dim_state,
            sampling_time,
            prediction_step_size,
            time_final=1,
            action_init=None,
            body_plan_init=None,
    ):
        print("simulator START")
        # init common params
        self.system = system
        self.dt = dt
        self.time_final = time_final
        self.dim_state = dim_state
        self.sys_out = system.out
        # init ros params
        rospy.Subscriber(A1_topic_name_req, Calf,
                         self.robotStateCallback, queue_size=1)
        self.ros_time = None
        self.ros_msg_time = None
        self.ros_state = None
        self.ros_MPC_params = None

        self.time_init = 0
        self.time_start = None
        self.mpc_params = None
        self.state = None
        self.prev_state = None
        self.time_now = 0
        self.system.receive_action(action_init, rospy.Time.now())
        self.system.receive_body_plan(body_plan_init)

        self.state_queue = deque(maxlen=prediction_step_size // sampling_time)

        print("simulator is READY")

    def do_sim_step(self):
        if (self.time_now >= self.time_final):
            print("simulation time is over")
            return -1

        while (self.ros_state is None):
            pass

        print("pub action")
        self.system.publishAction()
        print("action is published")
        self.state = self.ros_state
        self.time_now = self.ros_time - self.time_start
        self.observation = self.sys_out(self.state)
        self.msg_time = self.ros_msg_time
        self.mpc_params = self.ros_MPC_params
        self.ros_state = None
        self.ros_time = None
        self.ros_msg_time = None
        self.ros_MPC_params = None

    def robotStateCallback(self, robotStateMsg: Calf) -> None:
        self.ros_time = rospy.Time.now().to_sec()
        self.ros_msg_time = robotStateMsg.header.stamp
        self.ros_MPC_params = utils.MPCParams(robotStateMsg)
        self.ros_state = utils.Observation(
            robotStateMsg, self.ros_MPC_params).full_state

        # if self.prev_state is None:
        #     interp_state = self.ros_state
        # else:
        #     interp_state = self.interpolate_state(
        #         self.prev_state, self.ros_state)

        # self.prev_state = self.ros_state
        # self.ros_state = interp_state
        last_state = np.copy(self.ros_state)
        if len(self.state_queue) == 10:
            self.ros_state[24:36] = self.state_queue[0][24:36]
        self.state_queue.append(last_state)

        if self.time_start is None:
            self.time_start = self.ros_time + self.time_init

    def interpolate_state(self, state_prev, state_now):
        x = state_prev[:12]
        x_des = state_prev[24:36]
        return np.vstack([state_now[:24], self.interpolate(x, x_des)])

    def interpolate(self, x0, x1):
        x_interp = x0 + (x1 - x0)/0.03 * 0.003
        return x_interp

    def get_sim_step_data(self):
        time, state_full, observation, mpc_params, msg_time = (
            self.time_now,
            self.state,
            self.observation,
            self.mpc_params,
            self.msg_time
        )
        return time, state_full, observation, mpc_params, msg_time

import rospy
import time
import numpy as np


class OnlineScenarioA1():
    def __init__(
        self,
        simulator,
        controller,
        running_objective,
        predictor,
        state_init: np.ndarray = None,
        action_init: np.ndarray = None,
        observation_components_naming=[],
    ):
        print("scenario init")
        while (simulator.ros_state is None):
            rospy.loginfo_throttle_identical(
                0.5, "waiting for the first state")
            if rospy.is_shutdown():
                raise KeyboardInterrupt
        self.simulator = simulator
        self.system = simulator.system

        self.controller = controller
        # self.actor = controller.actor
        # self.critic = controller.critic
        self.running_objective = running_objective
        self.time_final = self.simulator.time_final

        self.last_time = time.time()
        self.state_init = state_init
        self.state_full = state_init
        self.action_init = action_init
        self.action = self.action_init
        self.observation = self.system.out(self.state_init)

        self.predictor = predictor
        self.sim_status = None
        self.time = 0
        self.time_old = 0
        self.delta_time = 0
        self.index = 0
        self.observation_components_naming = observation_components_naming
        print("scenario is ready")

    def run(self):
        print("RUN")
        while self.step():
            pass
        print("Episode ended successfully.")

    def step(self):
        # self.pre_step()
        print("do sim step")
        sim_status = self.simulator.do_sim_step()

        is_episode_ended = sim_status == -1
        if is_episode_ended:
            return False
        else:
            print("get_sim_step_data")
            (
                self.time,
                self.state_full,
                self.observation,
                self.mpc_params,
                msg_time,
            ) = self.simulator.get_sim_step_data()

            print("compute_action_sampled")
            # In future versions state vector being passed into controller should be obtained from an observer.
            time0 = time.time()
            self.action = self.controller.compute_action_sampled(
                time=self.time,
                state=self.state_full,
                observation=self.observation,
                mpc_params=self.mpc_params,
                observation_target=np.zeros(12),
                # observation_target=self.mpc_params.ref_body_plan.T
            )
            self.compute_time = time.time() - time0
            print("compute_body_plan")
            body_plan = self.predictor.predict(
                self.state_full, self.action.reshape((12, 1)))
            print("receive_action and body_plan")
            self.system.receive_action(self.action, msg_time)
            self.system.receive_body_plan(body_plan)
            print("post_step")
            # self.post_step()
            print("ACTION")
            print(self.action)
            print("log info")
            print(self.controller.critic.weights)
            self.printLogData()
            # self.sim_status = self.simulator.do_sim_step()
            return True

    def printLogData(self, **kwards):
        ros_time = rospy.Time.now().to_sec()
        time_now = self.simulator.time_now
        scenario_progress = time_now / self.simulator.time_final * 100
        dt = time.time() - self.last_time

        print("--------------{}-------------".format(self.index))
        print("real time:{:.3f}; rostime: {:.3f};  scenario progress: {:.3f}%".format(
            dt, time_now, scenario_progress))
        for key in kwards:
            print("{k}: {v:.3f}".format(k=key, v=(time.time() - kwards[key])))
        self.index += 1

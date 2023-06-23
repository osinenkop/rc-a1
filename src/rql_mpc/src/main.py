#!/home/catkin_ws/venv/bin/python
import traceback
import sys
import numpy as np

from system import A1system
from simulator import A1simulator
from controller import *
from critic import *

from actor import *
from predictor import *
from models import *
from optimizer import *
from scenario import *


def launch():
    p = {
        # GENERAL
        "final_time": 4,
        "prediction_horizon": 4,
        "prediction_step_size": 0.03,
        "sampling_time": 0.003,
        # System
        "dim_state": 12,
        "dim_input": 12,
        "dim_output": 12,
        # Predictor
        # "n_steps": 5,
        # actor optimizer
        "optimization_method": "ipopt",
        "opt_options": {
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1.0E-7,
            "ipopt.acceptable_obj_change_tol": 1.0E-4,
        },
        # init state and action
        "state_init": np.array([-0.02461464,  0.00024726,  0.26689154, -0.00379155, -0.03107476,        0.00093648,  0.00004392,  0.00000232, -0.00001977, -0.00006318,        0.00008691,  0.00000217]),
        "action_init": np.zeros(12),
        "body_plan_init": np.array([-0.02461332,   0.00024733,   0.26689095,  -0.00379344,        -0.03107215,   0.00093654,  -0.00780878,   0.00015397,         0.00573565,   0.0261033,   0.17923658,   0.00068405]),
        # MPC weights
        "QandR": [np.array([1., 1., 10., 1, 1, 1, 0.02, 0.02, 0.04, 0.01, 0.01, 0.002]) * 1000000,
                  [1e-4]]
    }

    system = A1system(
        A1_topic_name_res="/robot_1/calf/action",
        dim_state=p["dim_state"],
        dim_input=p["dim_input"],
        dim_output=p["dim_output"],
    )

    simulator = A1simulator(
        system=system,
        A1_topic_name_req="/robot_1/calf/observations",
        dim_state=p["dim_state"],
        sampling_time=p["sampling_time"],
        prediction_step_size=p["prediction_horizon"],
        time_final=p["final_time"],
        action_init=p["action_init"],
        body_plan_init=p["body_plan_init"],
    )

    predictor = EulerPredictorA1(
        pred_step_size=p["prediction_step_size"],
        system=system,
        dim_input=p["dim_input"],
        prediction_horizon=p["prediction_horizon"],
        # n_steps=p["n_steps"],
    )

    running_objective = RosOnlineQuadForm(
        weights=p["QandR"],
    )

    optimizer = CasADiOptimizer(
        opt_method=p["optimization_method"],
        opt_options=p["opt_options"],
    )

    critic_model = ModelQuadNoMixA1(
        dim_input=12,
        single_weight_min=0,
        single_weight_max=1e+6,
        weights_init=np.ones(12) * 10000,
    )

    critic = CriticOffPolicyBehaviour(
        dim_action=system.dim_input,
        dim_state=36,
        data_buffer_size=400,
        running_objective=running_objective,
        discount_factor=1,
        sampling_time=p["sampling_time"],
        state_init=p["state_init"],
        critic_regularization_param=0,
        batch_size=5,
        td_n=2,
        optimizer=optimizer,
        model=critic_model,
    )

    actor_model = ModelWeightContainer(
        system.dim_output, np.ones(system.dim_output))

    # actor = ActorMPCA1(
    #     dim_input=system.dim_input,
    #     dim_output=system.dim_output,
    #     prediction_horizon=predictor.prediction_horizon,
    #     predictor=predictor,
    #     optimizer=optimizer,
    #     critic=critic,
    #     running_objective=running_objective,
    #     model=actor_model,
    #     discount_factor=1,
    #     action_init=p["action_init"],
    #     state_init=p["state_init"],
    #     action_bounds=np.array([[-150, -150, -150]*4, [150, 150, 150]*4]).T
    # )

    actor = ActorRQLA1(
        dim_input=system.dim_input,
        dim_output=system.dim_output,
        prediction_horizon=predictor.prediction_horizon,
        predictor=predictor,
        optimizer=optimizer,
        critic=critic,
        running_objective=running_objective,
        model=actor_model,
        discount_factor=1,
        action_init=p["action_init"],
        state_init=p["state_init"],
        action_bounds=np.array([[-150, -150, -150]*4, [150, 150, 150]*4]).T
    )

    controller = RLController(
        critic_period=p["sampling_time"],
        time_start=simulator.time_start,
        is_fixed_critic_weights=False,  # IF you wana fix weights
        actor=actor,
        critic=critic,
    )

    scenario = OnlineScenarioA1(
        simulator=simulator,
        controller=controller,
        running_objective=running_objective,
        predictor=predictor,
        state_init=p["state_init"],
        action_init=p["action_init"],
    )
    scenario.run()

    print("DONE")
    return 0


if __name__ == "__main__":
    try:
        launch()
    except KeyboardInterrupt:
        passjob_results = launch()
    except Exception:
        print(traceback.format_exc())
        # or
        # print(sys.exc_info()[2])

import casadi as ca


class EulerPredictorA1():
    def __init__(
        self,
        pred_step_size: float,
        system,
        dim_input: int,
        prediction_horizon: int,
    ):
        self.system = system
        self.pred_step_size = pred_step_size
        self.compute_state_dynamics = system.compute_dynamics
        self.sys_out = system.out
        self.dim_input = dim_input
        self.prediction_horizon = prediction_horizon

    def predict(self, current_state_full, action):
        current_state = current_state_full[:12]
        next_state = current_state + self.pred_step_size * self.compute_state_dynamics(
            [], current_state_full, action
        )
        return next_state

    def predict_sequence(self, state_full, action_sequence):
        observation_sequence = ca.MX.zeros(
            self.dim_input, self.prediction_horizon)

        current_state = state_full
        for k in range(self.prediction_horizon):
            current_action = action_sequence[:, k]
            next_state = self.predict(current_state, current_action)
            observation_sequence[:, k] = ca.transpose(self.sys_out(next_state))
            current_state = ca.vcat(
                [next_state,
                 self.grf_positions_world[:, k+1],
                 self.ref_body_plan[:, k+1]
                 ])
        return observation_sequence

    def update_params(self, mpc_params):
        self.grf_positions_world = mpc_params.grf_positions_world.T
        self.ref_body_plan = mpc_params.ref_body_plan.T

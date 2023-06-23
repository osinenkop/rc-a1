import numpy as np
import casadi as ca
import time as timer
import scipy


class Actor:
    """
    Class of actors.
    These are to be passed to a `controller`.
    An `objective` (a loss) as well as an `optimizer` are passed to an `actor` externally.
    """

    def __call__(self, observation):
        """
        Return the most recent action taken by the actor.

        :param observation: Current observation of the system.
        :type observation: ndarray
        :returns: Most recent action taken by the actor.
        :rtype: ndarray
        """
        return self.action

    def reset(self):
        """
        Reset the actor to its initial state.
        """
        self.action_old = self.action_init
        self.action = self.action_init

    def __init__(
        self,
        dim_output: int = 2,
        dim_input: int = 5,
        prediction_horizon: int = 1,
        action_bounds=None,
        action_init=None,
        state_init=None,
        predictor=None,
        optimizer=None,
        critic=None,
        running_objective=None,
        model=None,
        discount_factor=1,
        epsilon_greedy=False,
        epsilon_greedy_parameter=0.0,
        observation_target=None,
    ):
        print("init actor")
        self.prediction_horizon = prediction_horizon
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.action_bounds = action_bounds
        self.optimizer = optimizer
        self.critic = critic
        self.running_objective = running_objective
        self.model = model

        self.predictor = predictor
        self.discount_factor = discount_factor
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_greedy_parameter = epsilon_greedy_parameter

        self.action_min = np.array(self.action_bounds)[:, 0]
        self.action_max = np.array(self.action_bounds)[:, 1]

        self.action_init = action_init
        self.action_old = action_init
        self.action_sequence_init = np.squeeze(
            np.tile(action_init, (1, self.prediction_horizon + 1)))

        self.action_sequence_min = np.squeeze(
            np.tile(self.action_min, (1, self.prediction_horizon + 1)))

        self.action_sequence_max = np.squeeze(
            np.tile(self.action_max, (1, self.prediction_horizon + 1)))

        self.action_bounds = np.array(
            [self.action_sequence_min, self.action_sequence_max]
        )
        self.action = self.action_old
        self.intrinsic_constraints = []
        if observation_target is None or observation_target == []:
            self.observation_target = np.zeros(self.dim_output)
        elif isinstance(observation_target, list):
            self.observation_target = np.array(observation_target)
        self.state_init = state_init
        self.observation_init = self.predictor.system.out(state_init)

    @property
    def weights(self):
        """
        Get the weights of the actor model.
        """
        return self.model.weights

    def update_target(self, observation_target):
        self.observation_target = observation_target

    def receive_observation(self, observation):
        """
        Update the current observation of the actor.
        :param observation: The current observation.
        :type observation: numpy array
        """
        self.observation = observation

    def receive_state(self, state):
        """
        Update the current observation of the actor.
        :param observation: The current observation.
        :type observation: numpy array
        """
        self.state = state

    def set_action(self, action):
        """
        Set the current action of the actor.
        :param action: The current action.
        :type action: numpy array
        """
        self.action_old = self.action
        self.action = action

    def update_action(self, observation=None):
        """
        Update the current action of the actor.
        :param observation: The current observation. If not provided, the previously received observation will be used.
        :type observation: numpy array, optional
        """
        self.action_old = self.action

        if observation is None:
            observation = self.observation

        self.action = self.model(observation - self.observation_target)

    def update_weights(self, weights=None):
        """
        Update the weights of the model of the actor.
        :param weights: The weights to update the model with. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        if weights is None:
            self.model.update_weights(self.optimized_weights)
        else:
            self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """
        Cache the current weights of the model of the actor.
        :param weights: The weights to cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        if weights is not None:
            self.model.cache_weights(weights)
        else:
            self.model.cache_weights(self.optimized_weights)

    def update_and_cache_weights(self, weights=None):
        """
        Update and cache the weights of the model of the actor.
        :param weights: The weights to update and cache. If not provided, the previously optimized weights will be used.
        :type weights: numpy array, optional
        """
        self.update_weights(weights)
        self.cache_weights(weights)

    def restore_weights(self):
        """
        Restore the previously cached weights of the model of the actor.
        """
        self.model.restore_weights()
        self.set_action(self.action_old)

    def accept_or_reject_weights(
        self, weights, constraint_functions=None, optimizer_engine="SciPy", atol=1e-5
    ):
        if constraint_functions is None:
            constraints_not_violated = True
        else:
            not_violated = [
                cond(weights) <= atol for cond in constraint_functions]
            constraints_not_violated = all(
                [np.all(condition) for condition in not_violated])
            # print(not_violated)

        if constraints_not_violated:
            return "accepted"
        else:
            return "rejected"

    def optimize_weights(self, constraint_functions=None, time=None):
        """
        Method to optimize the current actor weights. The old (previous) weights are stored.
        The `time` argument is used for debugging purposes.
        If weights satisfying constraints are found, the method returns the status `accepted`.
        Otherwise, it returns the status `rejected`.

        :param constraint_functions: List of functions defining constraints on the optimization.
        :type constraint_functions: list of callables, optional
        :param time: Debugging parameter to track time during optimization process.
        :type time: float, optional
        :returns: String indicating whether the optimization process was accepted or rejected.
        :rtype: str
        """
        final_count_of_actions = self.prediction_horizon + 1
        action_sequence = np.squeeze(
            np.tile(self.action, (1, final_count_of_actions)))
        action_sequence_init_reshaped = np.reshape(
            action_sequence,
            [final_count_of_actions * self.dim_output],
        )

        constraints = []

        if self.optimizer.engine == "CasADi":
            action_sequence_init_reshaped = ca.DM(
                action_sequence_init_reshaped)

            symbolic_var = ca.MX.sym(
                "x",
                action_sequence_init_reshaped.shape,
            )

            # def actor_objective(action_sequence): return self.objective(
            #     action_sequence, self.observation
            # )

            # actor_objective = rc.lambda2symb(actor_objective, symbolic_var)
            actor_objective = self.objective(symbolic_var, self.observation)

            constraint_functions = []
            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var, self.observation
                )

            if self.intrinsic_constraints:
                intrisic_constraints = [
                    constraint(symbolic_var)
                    for constraint in self.intrinsic_constraints
                ]
            else:
                intrisic_constraints = []

            self.optimized_weights = self.optimizer.optimize(
                actor_objective,
                action_sequence_init_reshaped,
                self.action_bounds,
                constraints=intrisic_constraints + constraint_functions,
                decision_variable_symbolic=symbolic_var,
            )
            # self.cost_function = actor_objective
            # self.constraint = intrisic_constraints[0]
            # self.weights_init = action_sequence_init_reshaped
            # self.symbolic_var = symbolic_var

        if self.intrinsic_constraints:
            # DEBUG ==============================
            # print("with constraint functions")
            # /DEBUG =============================
            self.weights_acceptance_status = self.accept_or_reject_weights(
                self.optimized_weights,
                constraint_functions=self.intrinsic_constraints,
                optimizer_engine=self.optimizer.engine,
            )
        else:
            # DEBUG ==============================
            # print("without constraint functions")
            # /DEBUG =============================
            self.weights_acceptance_status = "accepted"

        return self.weights_acceptance_status


class ActorMPCA1(Actor):

    def objective(
        self,
        action_sequence,
        observation_full,
    ):
        """
        Calculates the actor objective for the given action sequence and observation using Rollout Q-learning (RQL).

        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_output) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_output,) representing the current observation
        :type observation: numpy.ndarray
        :return: actor objective for the given action sequence and observation
        :rtype: float
        """

        action_sequence_reshaped = ca.reshape(
            action_sequence, self.dim_output, self.prediction_horizon + 1
        )

        print(action_sequence_reshaped.shape)

        observation = observation_full[:12]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation_full, action_sequence_reshaped
        )

        observation_sequence = ca.hcat(
            (
                observation,
                observation_sequence_predicted,
            )
        )

        actor_objective = 0

        for k in range(self.prediction_horizon + 1):
            full_state = ca.vcat([
                observation_sequence[:, k],
                self.grf_positions_world[:, k],
                self.ref_body_plan[:, k],
            ])
            actor_objective += self.discount_factor**k * self.running_objective(
                full_state, action_sequence_reshaped[:, k]
            )
        return actor_objective

    def update_constraints(self, MPC_params):
        self.contact_schedule = MPC_params.contact_schedule
        self.intrinsic_constraints = (
            [
                self.constraint_for_contact_schedule,
                self.anti_constraint_for_contact_schedule,
                self.constraint_for_friction_cone,
            ]
        )

    def constraint_for_contact_schedule(self, action):
        u = action
        contacts = self.contact_schedule
        u_shape = u.shape[0]
        u_dim = 12
        N = u_shape // u_dim
        contact_constraints = []
        for n in range(N):

            n_contact_constraints = np.kron(
                np.diag(1 - contacts[n, :]), np.eye(3))
            contact_constraints.append(n_contact_constraints)
        contact_constraints = scipy.linalg.block_diag(*contact_constraints)
        return contact_constraints @ u

    def constraint_for_friction_cone(self, action):
        u = action
        u_shape = u.shape[0]
        u_dim = 12
        N = u_shape // u_dim
        mu = 0.3
        cone = np.array([[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu]])
        friction_constraints = np.kron(np.eye(4), cone)
        contact_constraints = []
        for n in range(N):
            contact_constraints.append(friction_constraints)

        constraints = scipy.linalg.block_diag(*contact_constraints)
        return constraints @ u

    def anti_constraint_for_contact_schedule(self, action):
        return -self.constraint_for_contact_schedule(action)

    def update_mpc_params(self, mpc_params):
        self.grf_positions_world = mpc_params.grf_positions_world.T
        self.ref_body_plan = mpc_params.ref_body_plan.T
        self.predictor.update_params(mpc_params)
        self.update_constraints(mpc_params)


class ActorRQLA1(ActorMPCA1):

    def objective(
        self,
        action_sequence,
        observation_full,
    ):
        """
        Calculates the actor objective for the given action sequence and observation using Rollout Q-learning (RQL).

        :param action_sequence: numpy array of shape (prediction_horizon+1, dim_output) representing the sequence of actions to optimize
        :type action_sequence: numpy.ndarray
        :param observation: numpy array of shape (dim_output,) representing the current observation
        :type observation: numpy.ndarray
        :return: actor objective for the given action sequence and observation
        :rtype: float
        """
        action_sequence_reshaped = ca.reshape(
            action_sequence, self.dim_output, self.prediction_horizon + 1
        )

        observation = observation_full[:12]
        observation_sequence = [observation]

        observation_sequence_predicted = self.predictor.predict_sequence(
            observation_full, action_sequence_reshaped
        )

        observation_sequence = ca.hcat(
            (
                observation,
                observation_sequence_predicted,
            )
        )

        actor_objective = 0

        for k in range(self.prediction_horizon):
            state = observation_sequence[:, k]
            full_state = ca.vcat([
                state,
                self.grf_positions_world[:, k],
                self.ref_body_plan[:, k],
            ])
            actor_objective += self.discount_factor**k * self.running_objective(
                full_state, action_sequence_reshaped[:, k]
            )

        # actor_objective += self.critic(
        #     observation_sequence[:, -1] - self.observation_target,
        #     action_sequence_reshaped[:, -1],
        #     use_stored_weights=True,
        # )

        last_state_full = ca.vcat([
            observation_sequence[:, k+1],
            self.grf_positions_world[:, k+1],
            self.ref_body_plan[:, k+1],
        ]
            # self.interpolate(self.ref_body_plan[:, k+1], self.ref_body_plan[:, k+2])]
        )

        actor_objective += self.critic(
            last_state_full,
            action_sequence_reshaped[:, -1],
            use_stored_weights=True,
        )

        return actor_objective

    def interpolate(self, x0, x1):
        x_interp = x0 + (x1 - x0)/0.03 * 0.003
        return x_interp

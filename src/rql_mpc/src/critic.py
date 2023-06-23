
import numpy as np
import casadi as ca
import random


class Critic():
    """
    Critic base class.

    A critic is an object that estimates or provides the value of a given action or state in a reinforcement learning problem.

    The critic estimates the value of an action by learning from past experience, typically through the optimization of a loss function.
    """

    def __init__(
        self,
        dim_action: int,
        dim_state: int,
        data_buffer_size: int,
        state_init: np.ndarray = None,
        optimizer=None,
        model=None,
        running_objective=None,
        discount_factor: float = 1.0,
        observation_target=None,
        sampling_time: float = 0.01,
        critic_regularization_param: float = 0.0,
    ):
        print("critic init")

        self.data_buffer_size = data_buffer_size
        self.system_dim_input = dim_action
        self.system_dim_output = dim_state

        self.optimizer = optimizer
        self.model = model

        self.initialize_buffers()

        if observation_target is None or observation_target == []:
            observation_target = np.zeros(dim_state)
        elif isinstance(observation_target, list):
            self.observation_target = ca.array(observation_target)

        self.discount_factor = discount_factor
        self.running_objective = running_objective

        self.current_critic_loss = 0
        self.outcome = 0
        self.sampling_time = sampling_time
        self.intrinsic_constraints = []
        self.penalty_param = 0
        self.critic_regularization_param = critic_regularization_param
        self.state_init = state_init
        print("critic is done")

    def update_target(self, observation_target):
        self.observation_target = observation_target

    def receive_state(self, state):
        self.state = state

    def __call__(self, *args, use_stored_weights=False):
        """
        Compute the value of the critic function for a given observation and/or action.

        :param args: tuple of the form (observation, action) or (observation,)
        :type args: tuple
        :param use_stored_weights: flag indicating whether to use the stored weights of the critic model or the current weights
        :type use_stored_weights: bool
        :return: value of the critic function
        :rtype: float
        """

        if len(args) == 2:
            chi = ca.vcat(args)
        else:
            chi = args[0]
        return self.model(chi, use_stored_weights=use_stored_weights)

    @property
    def weights(self):
        """
        Get the weights of the critic model.
        """
        return self.model.weights

    def update_weights(self, weights=None):
        """
        Update the weights of the critic model.

        :param weights: new weights to be used for the critic model, if not provided the optimized weights will be used
        :type weights: numpy array
        """
        if weights is None:
            self.model.update_weights(self.optimized_weights)
        else:
            self.model.update_weights(weights)

    def cache_weights(self, weights=None):
        """
        Stores a copy of the current model weights.

        :param weights: An optional ndarray of weights to store. If not provided, the current
            model weights are stored. Default is None.
        """
        if weights is not None:
            self.model.cache_weights(weights)
        else:
            self.model.cache_weights(self.optimized_weights)

    def restore_weights(self):
        """
        Restores the model weights to the cached weights.
        """
        self.model.restore_weights()

    def update_and_cache_weights(self, weights=None):
        """
        Update the model's weights and cache the new values.

        :param weights: new weights for the model (optional)
        """
        self.update_weights(weights)
        self.cache_weights(weights)

    def accept_or_reject_weights(
        self, weights, constraint_functions=None, optimizer_engine="SciPy", atol=1e-10
    ):

        if constraint_functions is None:
            constraints_not_violated = True
        else:
            not_violated = [
                cond(weights) <= atol for cond in constraint_functions]
            constraints_not_violated = all(not_violated)

        if constraints_not_violated:
            return "accepted"
        else:
            return "rejected"

    def optimize_weights(
        self,
        time=None,
    ):
        """
        Compute optimized critic weights, possibly subject to constraints.
        If weights satisfying constraints are found, the method returns the status `accepted`.
        Otherwise, it returns the status `rejected`.

        :param time: optional time parameter for use in CasADi and SciPy optimization.
        :type time: float, optional
        :return: acceptance status of the optimized weights, either `accepted` or `rejected`.
        :rtype: str
        """

        self.optimized_weights = self._CasADi_update(
            self.intrinsic_constraints)

        if self.intrinsic_constraints != []:
            # print("with constraint functions")
            self.weights_acceptance_status = self.accept_or_reject_weights(
                self.optimized_weights,
                optimizer_engine=self.optimizer.engine,
                constraint_functions=self.intrinsic_constraints,
            )
        else:
            # print("without constraint functions")
            self.weights_acceptance_status = "accepted"

        return self.weights_acceptance_status

    def update_buffers(self, observation, action):
        """
        Updates the buffers of the critic with the given observation and action.

        :param observation: the current observation of the system.
        :type observation: np.ndarray
        :param action: the current action taken by the actor.
        :type action: np.ndarray
        """
        self.action_buffer = np.column_stack(
            [self.action_buffer[:, 1:], np.array(action)]
        )
        self.observation_buffer = np.column_stack(
            [self.observation_buffer[:, 1:],
             np.array(observation)],
        )
        self.update_outcome(observation, action)

        self.current_observation = observation
        self.current_action = action

    def initialize_buffers(self):
        """
        Initialize the action and observation buffers with zeros.
        """
        self.action_buffer = np.zeros(
            (int(self.system_dim_input), int(self.data_buffer_size)))

        self.observation_buffer = np.zeros(
            (int(self.system_dim_output), int(self.data_buffer_size)))

    def update_outcome(self, observation, action):
        """
        Update the outcome variable based on the running objective and the current observation and action.
        :param observation: current observation
        :type observation: np.ndarray
        :param action: current action
        :type action: np.ndarray
        """

        self.outcome += self.running_objective(
            observation, action) * self.sampling_time

    def _CasADi_update(self, intrinsic_constraints=None):

        weights_init = ca.DM(self.model.cache.weights)
        symbolic_var = ca.MX.sym("x", np.shape(weights_init))

        constraints = ()
        weight_bounds = [self.model.weight_min, self.model.weight_max]
        data_buffer = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        def cost_function(weights): return self.objective(
            data_buffer, weights=weights)

        cost_function = cost_function(symbolic_var)

        is_penalty = int(self.penalty_param > 0)

        if intrinsic_constraints:
            constraints = [
                constraint(symbolic_var)
                for constraint in intrinsic_constraints[is_penalty:]
            ]

        optimized_weights = self.optimizer.optimize(
            cost_function,
            weights_init,
            weight_bounds,
            constraints=constraints,
            decision_variable_symbolic=symbolic_var,
        )

        self.cost_function = cost_function
        self.constraint = constraints
        self.weights_init = weights_init
        self.symbolic_var = symbolic_var
        print(optimized_weights)
        return optimized_weights


class CriticOffPolicyBehaviour(Critic):
    def __init__(self, *args, batch_size, td_n, **kwargs):
        print("critic init")
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.td_n = td_n

        self.n_buffer_updates = 0

    """
    This is the class of critics that are represented as functions of observation only.
    """

    def reset(self):
        super().reset()
        self.n_buffer_updates = 0

    def update_buffers(self, observation, action):
        super().update_buffers(observation, action)
        self.n_buffer_updates += 1

    def update_and_cache_weights(self, weights=None):
        if self.is_enough_valid_elements_in_buffer():
            super().update_and_cache_weights(weights)

    def optimize_weights(self, time=None):
        if self.is_enough_valid_elements_in_buffer():
            print("___WEIGHTS ARE UPDATING___")
            super().optimize_weights(time)

    def get_first_valid_idx_in_buffer(self):
        return max(self.data_buffer_size - self.n_buffer_updates, 0)

    def is_enough_valid_elements_in_buffer(self):
        return (
            self.data_buffer_size - self.get_first_valid_idx_in_buffer()
            >= self.td_n + self.batch_size + 1
        )

    def get_batch_ids(self):
        if not self.is_enough_valid_elements_in_buffer():
            raise ("Not enough valid elements in buffer for critic objective call")

        buffer_idx_for_latest_td_term = self.data_buffer_size - self.td_n - 2
        if self.batch_size == 1:
            batch_ids = np.array([buffer_idx_for_latest_td_term])
        elif (
            buffer_idx_for_latest_td_term - self.get_first_valid_idx_in_buffer()
            == self.batch_size - 1
        ):
            batch_ids = np.arange(
                self.get_first_valid_idx_in_buffer(), buffer_idx_for_latest_td_term + 1
            )
        else:
            sampled_ids = random.sample(
                range(
                    self.get_first_valid_idx_in_buffer(), buffer_idx_for_latest_td_term
                ),
                self.batch_size - 1,
            )
            batch_ids = np.hstack([sampled_ids, buffer_idx_for_latest_td_term])

        return batch_ids

    def objective(self, data_buffer=None, weights=None):
        """
        Compute the objective function of the critic, which is typically a squared temporal difference.
        :param data_buffer: a dictionary containing the action and observation buffers, if different from the class attributes.
        :type data_buffer: dict, optional
        :param weights: the weights of the critic model, if different from the stored weights.
        :type weights: numpy.ndarray, optional
        :return: the value of the objective function
        :rtype: float
        """
        if data_buffer is None:
            observation_buffer = self.observation_buffer
            action_buffer = self.action_buffer
        else:
            observation_buffer = data_buffer["observation_buffer"]
            action_buffer = data_buffer["action_buffer"]

        batch_ids = self.get_batch_ids()

        # Calculation of critic objective
        critic_objective = 0
        for buffer_idx in batch_ids:
            temporal_difference = 0

            temporal_difference += self.model(
                observation_buffer[:, buffer_idx],
                action_buffer[:, buffer_idx + 1],
                weights=weights,
            )

            for td_n_idx in range(self.td_n):
                temporal_difference -= (
                    self.discount_factor**td_n_idx
                    * self.running_objective(
                        observation_buffer[:, buffer_idx + td_n_idx],
                        action_buffer[:, buffer_idx + td_n_idx + 1],
                    )
                    * self.sampling_time
                )

            temporal_difference -= self.discount_factor**self.td_n * self.model(
                observation_buffer[:, buffer_idx + self.td_n],
                action_buffer[:, buffer_idx + self.td_n + 1],
                use_stored_weights=True,
            )

            if self.critic_regularization_param > 0:
                weights_current = weights
                weights_last_good = self.model.cache.weights
                regularization_term = (
                    ca.sum1((weights_current - weights_last_good)**2)
                    * self.critic_regularization_param
                )
            else:
                regularization_term = 0

            critic_objective += 1 / 2 * temporal_difference**2 / \
                self.batch_size + regularization_term
        return critic_objective

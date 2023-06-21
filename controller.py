import rospy
import numpy as np
import casadi as ca
import time as timer


class A1myMPC():
    def __init__(self, sampling_time: float, horizon: int, diff_step: float, system, predictor) -> None:
        print("mpc controller init")
        self.time_start = rospy.Time.now().to_sec()
        self.first_run = True
        self.action_old = None
        self.predictor = predictor
        self.creat_MPC_solver(horizon, diff_step, system)
        print("mpc controller is ready")

    def compute_action_sampled(self, time, state, observation, mpc_params, observation_target=[]):
        """
        Compute sampled action.

        """
        self.predictor.update_params(mpc_params)

        action = self.compute_action(state, observation, mpc_params)
        return action

    def compute_action(self, state, observation, parameters):
        N = self.N
        x0_real = observation[:self.x_dim]
        x_ref_real = parameters.ref_body_plan[0:N, :].reshape((-1))
        x_foots_real = parameters.grf_positions_world[0:N, :].reshape(-1)
        contacts_real = parameters.contact_schedule[0:N, :]
        mpc_res = self.mpc(x0_real, x_ref_real, x_foots_real, contacts_real)

        result = np.vstack([mpc_res[0][:self.u_dim], mpc_res[1][:self.x_dim]])
        # print(result)
        action = mpc_res[0][:self.u_dim].full()

        return action

    def creat_MPC_solver(self, horizon: int, diff_step: float, system):
        opti = ca.Opti()

        x_dim = 12
        self.x_dim = x_dim
        u_dim = 12
        self.u_dim = u_dim

        foot_num = 4

        N = horizon
        self.N = N

        dt = diff_step
        mu = 0.3
        rhs = system.compute_dynamics

        def euler_integrator(x0, u, p, dt):
            return x0 + dt * rhs(time=None, full_state=ca.vcat([x0, p, ca.vcat([0]*12)]), action=u)

        x = opti.variable(x_dim * N)
        u = opti.variable(u_dim * (N - 1))
        x0 = opti.parameter(x_dim)
        x_ref = opti.parameter(x_dim * N)
        x_foots = opti.parameter(3 * foot_num * N)
        contacts = opti.parameter(N, foot_num)

        # friction cone
        cone = np.array([[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu]])
        friction_constraints = ca.kron(np.eye(4), cone)

        u_ref = ca.MX.zeros(u_dim * (N-1))  # np.zeros(u_dim * (N - 1))
        for i in range(N-1):
            for leg in range(foot_num):
                u_ref[i * u_dim + leg * 3 + 2] = 13.74 * \
                    9.81 / ca.sum2(contacts[i, :])
        # TO DO fix count of contacts in u_ref

        dx = x - x_ref
        du = u - u_ref

        weights = [1., 1., 10., 1, 1, 1,
                   0.02, 0.02, 0.04, 0.01, 0.01, 0.002]

        Q = np.diag(weights * N)
        R = np.diag([2.5e-08] * u_dim * (N - 1))

        weighted_sum = dx.T @ Q @ dx + du.T @ R @ du
        opti.minimize(weighted_sum)

        opti.subject_to(x[0:x_dim] == x0)
        for n in range(N - 1):
            opti.subject_to(x[(n + 1) * x_dim:(n + 2) *
                            x_dim] == euler_integrator(x[n * x_dim:(n + 1) * x_dim],
                                                       u[n *
                                                           u_dim:(n + 1) * u_dim],
                                                       x_foots[n * u_dim:(n + 1) *
                                                               u_dim],
                                                       dt=dt))

            contact_constraints = ca.kron(
                ca.diag(1 - contacts[n, :]), np.eye(3))
            opti.subject_to(contact_constraints @
                            u[n * u_dim:(n + 1) * u_dim] == 0)

            opti.subject_to(friction_constraints @
                            u[n * u_dim:(n + 1) * u_dim] <= 0)

        opti_options = {'print_in': False,
                        'print_out': False, 'print_time': False}
        solver_options = {'print_level': 0}

        opti.solver('ipopt', opti_options, solver_options)

        self.mpc = opti.to_function("MPC", [x0, x_ref, x_foots, contacts], [u, x], [
                                    "x0", "x_ref", "x_foots", "contacts"], ["u_opt", "x"])


class RLController():
    """
    Reinforcement learning controller class.
    Takes instances of `actor` and `critic` to operate.
    Action computation is sampled, i.e., actions are computed at discrete, equi-distant moments in time.
    `critic` in turn is updated every `critic_period` units of time.
    """

    def __init__(
        self,
        critic_period=0.1,
        actor=None,
        critic=None,
        time_start=0,
        action_bounds=None,
        sampling_time: float = 0.1,
        is_fixed_critic_weights: bool = False,
    ):
        self.actor = actor
        self.critic = critic
        self.action_bounds = action_bounds

        self.critic_clock = time_start
        self.critic_period = critic_period
        self.weights_difference_norms = []
        self.sampling_time = sampling_time
        self.observation_target = []
        self.is_fixed_critic_weights = is_fixed_critic_weights
        self.comp_time = 0

    def compute_action_sampled(
        self, time, state, observation, mpc_params, constraints=(), observation_target=[]
    ):
        self.actor.update_mpc_params(mpc_params)
        self.observation_target = observation_target
        is_time_for_critic_update = True

        is_critic_update = (
            is_time_for_critic_update and not self.is_fixed_critic_weights
        )

        self.compute_action(
            state,
            observation,
            time=time,
            is_critic_update=is_critic_update,
            observation_target=observation_target,
        )
        return self.actor.action

    def compute_action(
        self, state_full, observation_full, is_critic_update=True, time=0, observation_target=[]
    ):
        # This method is called to generate an event in order for callbacks to fire.
        # Namely, we want to save the critic weights, for instance, and other stuff.
        self.pre_compute_action()

        # Store current action and observation in critic's data buffer
        self.critic.update_buffers(observation_full, self.actor.action)
        self.critic.receive_state(state_full)
        # Store current observation in actor

        self.actor.receive_observation(observation_full)
        self.actor.receive_state(state_full)

        self.actor.update_target(observation_target)
        self.critic.update_target(observation_target)

        if is_critic_update:
            # Optimize critic's model weights
            self.critic.optimize_weights(time=time)
            # Substitute and cache critic's model optimized weights
            self.critic.update_and_cache_weights()
        tic = timer.perf_counter()
        # Optimize actor's model weights based on current observation
        self.actor.optimize_weights()
        # Substitute and cache weights in the actor's model
        self.actor.update_and_cache_weights()
        self.actor.update_action(observation_full)
        toc = timer.perf_counter()
        self.comp_time = toc-tic

        return self.actor.action

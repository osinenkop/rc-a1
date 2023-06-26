import rospy
import numpy as np
import casadi as ca
import time as timer
import random
import scipy


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

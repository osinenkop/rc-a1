import numpy as np
import casadi as ca
from copy import deepcopy


class Model():
    """
    Blueprint of a model.
    """

    def __call__(self, *args, weights=None, use_stored_weights=False):
        if use_stored_weights is False:
            if weights is not None:
                return self.forward(*args, weights=weights)
            else:
                return self.forward(*args, weights=self.weights)
        else:
            return self.cache.forward(*args, weights=self.cache.weights)

    def update_weights(self, weights):
        self.weights = weights

    def cache_weights(self, weights=None):
        if "cache" not in self.__dict__.keys():
            self.cache = deepcopy(self)

        if weights is None:
            self.cache.weights = self.weights
        else:
            self.cache.weights = weights

    def update_and_cache_weights(self, weights):
        self.cache_weights(weights)
        self.update_weights(weights)

    def restore_weights(self):

        self.update_and_cache_weights(self.cache.weights)


class ModelQuadNoMix(Model):
    """
    Quadratic model (no mixed terms).

    """

    model_name = "quad-nomix"

    def __init__(
        self,
        dim_input,
        single_weight_min=1e-6,
        single_weight_max=1e3,
        weights_init=None,
    ):
        print("quad no mix")
        self.dim_weights = dim_input
        self.weight_min = single_weight_min * np.ones(self.dim_weights)
        self.weight_max = single_weight_max * np.ones(self.dim_weights)
        if (weights_init is None):
            self.weights_init = (self.weight_min + self.weight_max) / 2.0
        else:
            self.weights_init = weights_init
        self.weights = self.weights_init
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            vec = ca.vcat(argin)
        else:
            vec = argin[0]

        if isinstance(vec, tuple):
            vec = vec[0]

        polynom = vec * vec

        result = ca.dot(weights, polynom)
        return result

    def update_weights(self, weights):
        self.weights = weights

    def cache_weights(self, weights=None):
        if "cache" not in self.__dict__.keys():
            self.cache = deepcopy(self)

        if weights is None:
            self.cache.weights = self.weights
        else:
            self.cache.weights = weights

    def update_and_cache_weights(self, weights):
        self.cache_weights(weights)
        self.update_weights(weights)


class ModelQuadNoMixA1(ModelQuadNoMix):

    def forward(self, *argin, weights=None):
        if len(argin) > 1:
            input = ca.vcat(argin)
        else:
            input = argin[0]
        if (input.shape[0] != 48):
            raise Exception("input.shape[0] != 48 in model")
        state_full = input[:36]
        action = input[36:48]
        dx, df, df_m = self.get_dx_df_dfm(state_full, action)
        return super().forward(dx, weights=weights)

    def get_dx_df_dfm(self, observation, action):
        forces = action.T.reshape((3, 4))
        sum_forces = ca.sum2(forces)
        ideal_forces = 13.74 * np.array([0, 0, 9.81])
        df = sum_forces - ideal_forces

        foot_p = observation[12:24].T.reshape((3, 4))
        body_pose = observation[:3]

        sum_forces_m = 0
        for i in range(4):
            sum_forces_m += ca.cross(foot_p[:, i]-body_pose, forces[:, i])
        df_m = sum_forces_m

        dx = observation[:12] - observation[24:36]
        return dx, df, df_m


class RosOnlineQuadForm(Model):
    """
    Quadratic form.

    """

    def __init__(self, weights=None, get_new_obj_cfgs=None):
        print("Ros quad form init")
        self.weights = weights
        self.u_dim = 12
        print("Ros quad form is ready")

    def forward(self, observation_full, action, weights=None):

        if (observation_full.shape[0] != 36):
            raise Exception("forward error")
        if (action.shape[0] != 12):
            raise Exception("forward error")

        x0 = observation_full[:12]
        x1 = observation_full[24:36]

        dx = x0 - x1
        u0_ref = np.ones((12, 1)) * 13.74 * 9.81 / 4
        # u0_ref = np.zeros((12, 1))
        du = action - u0_ref

        Q = np.diag(weights[0])
        R = np.diag(weights[1] * self.u_dim)

        # R = np.diag([2.5e-08] * self.u_dim)

        objective = dx.T @ Q @ dx / 2 + du.T @ R @ du / 2
        return objective[0, 0]


class ModelWeightContainer(Model):
    """
    Trivial model, which is typically used in actor in which actions are being optimized directly.

    """

    model_name = "action-sequence"

    def __init__(self, dim_output, weights_init=None):
        print("ModelWeightContainer init")
        self.dim_output = dim_output
        self.weights = weights_init
        self.weights_init = weights_init
        self.update_and_cache_weights(self.weights)

    def forward(self, *argin, weights=None):
        return weights[: self.dim_output]

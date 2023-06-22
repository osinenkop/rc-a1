from casadi import vertcat, nlpsol, DM, MX, Function


class CasADiOptimizer():
    engine = "CasADi"

    def __init__(self, opt_method, opt_options, verbose=False):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    def optimize(
        self,
        objective,
        initial_guess,
        bounds,
        constraints=(),
        decision_variable_symbolic=None,
    ):

        optimization_problem = {
            "f": objective,
            "x": vertcat(decision_variable_symbolic),
            "g": vertcat(*constraints),
        }

        atol = 1e-10
        if isinstance(constraints, (tuple, list)):
            upper_bound_constraint = [atol for _ in constraints]
        elif isinstance(constraints, (MX, DM, int, float)):
            upper_bound_constraint = [atol]

        try:
            solver = nlpsol(
                "solver",
                self.opt_method,
                optimization_problem,
                self.opt_options,
            )
        except Exception as e:
            print(e)
            return initial_guess

        upper_bound_constraint = [0]
        if upper_bound_constraint is not None and len(upper_bound_constraint) > 0:
            result = solver(
                x0=initial_guess,
                lbx=bounds[0],
                ubx=bounds[1],
                ubg=upper_bound_constraint,
            )
        else:
            result = solver(x0=initial_guess, lbx=bounds[0], ubx=bounds[1])

        # DEBUG
        # g1 = Function("g1", [symbolic_var], [constraints])

        # print(g1(result["x"]))
        # DEBUG

        return result["x"].T.full().flatten()

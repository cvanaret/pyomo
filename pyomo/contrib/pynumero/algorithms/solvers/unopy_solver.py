# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""
The unopy_solver module includes two solvers that call unopy (the Python interface
to the Uno solver). One, UnopySolver, is a solver that operates on a
CyIpoptProblemInterface (such as CyIpoptNLP) by translating it into a unopy Model.
The other, PyomoUnopySolver, operates directly on a Pyomo model.

Uno unifies Lagrange-Newton methods (SQP and interior-point). By default the solver
is configured with the "ipopt" preset so that it closely mimics IPOPT behaviour.
"""

import sys
import logging

from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.common.tee import capture_output
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.objective import Objective

# Because pynumero.interfaces requires numpy, we will leverage deferred
# imports here so that the solver can be registered even when numpy is
# not available.
pyomo_nlp = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_nlp")[0]
pyomo_grey_box = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp")[
    0
]
egb = attempt_import("pyomo.contrib.pynumero.interfaces.external_grey_box")[0]

# Defer this import so that importing this module (PyomoCyIpoptSolver in
# particular) does not rely on an attempted cyipopt import.
cyipopt_interface, _ = attempt_import(
    "pyomo.contrib.pynumero.interfaces.cyipopt_interface"
)

from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import Block, Objective, minimize
from pyomo.opt import SolverStatus, SolverResults, TerminationCondition
from pyomo.opt.results.solution import Solution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping from unopy solution/optimization statuses to Pyomo TerminationCondition
# ---------------------------------------------------------------------------

def uno_status_to_termination_condition(optimization_status, solution_status):
	if optimization_status == "UNO_ITERATION_LIMIT":
		return TerminationCondition.maxIterations
	elif optimization_status == "UNO_TIME_LIMIT":
		return TerminationCondition.maxTimeLimit
	elif optimization_status == "UNO_EVALUATION_ERROR":
		return TerminationCondition.internalSolverError
	elif optimization_status == "UNO_ALGORITHMIC_ERROR":
		return TerminationCondition.solverFailure
	elif optimization_status == "UNO_USER_TERMINATION":
		return TerminationCondition.userInterrupt
	else: # "UNO_SUCCESS"
		if solution_status == "UNO_FEASIBLE_KKT_POINT":
			return TerminationCondition.optimal
		elif solution_status == "UNO_FEASIBLE_FJ_POINT":
			return TerminationCondition.feasible
		elif solution_status == "UNO_INFEASIBLE_STATIONARY_POINT":
			return TerminationCondition.infeasible
		elif solution_status == "UNO_FEASIBLE_SMALL_STEP":
			return TerminationCondition.minStepLength
		elif solution_status == "UNO_INFEASIBLE_SMALL_STEP":
			return TerminationCondition.minStepLength
		elif solution_status == "UNO_UNBOUNDED":
			return TerminationCondition.unbounded
		elif solution_status == "UNO_NOT_OPTIMAL":
			return TerminationCondition.noSolution

def _uno_termination_condition(result):
    """Derive the Pyomo TerminationCondition from a unopy result object."""
    opt_status = str(result.optimization_status)
    sol_status = str(result.solution_status)
    return uno_status_to_termination_condition(opt_status, sol_status)


# ---------------------------------------------------------------------------
# Helper: build a unopy.Model from a CyIpoptNLP-compatible problem interface
# ---------------------------------------------------------------------------

def _build_unopy_model(problem):
    """
    Translate a CyIpoptProblemInterface (e.g. CyIpoptNLP) into a unopy.Model.

    The CyIpoptProblemInterface exposes:
      x_lb(), x_ub()              – variable bounds
      g_lb(), g_ub()              – constraint bounds
      objective(x)                – objective value (scalar return)
      gradient(x)                 – objective gradient (dense array return)
      constraints(x)              – constraint vector (array return)
      jacobianstructure()         – (rows, cols) sparsity pattern
      jacobian(x)                 – Jacobian values (sparse array return)
      hessianstructure()          – (rows, cols) sparsity pattern (lower triangle)
      hessian(x, lam, obj_factor) – Hessian values (sparse array return)
      x_init()                    – initial primal point

    Note on callback conventions
    -----------------------------
    unopy callbacks mutate their output array in-place and do not return a value,
    which differs from the cyipopt convention of returning a new array.  The
    wrappers below bridge this gap.  The unopy Hessian callback also uses the
    argument names ``objective_multiplier`` / ``multipliers`` rather than
    ``obj_factor`` / ``lam``.

    The Lagrangian sign convention is set to ``unopy.MULTIPLIER_POSITIVE`` to
    match the IPOPT / cyipopt convention used by CyIpoptNLP.
    """
    x_lb = np.array(problem.x_lb(), dtype=np.float64)
    x_ub = np.array(problem.x_ub(), dtype=np.float64)
    n_vars = len(x_lb)

    g_lb = np.array(problem.g_lb(), dtype=np.float64)
    g_ub = np.array(problem.g_ub(), dtype=np.float64)
    n_con = len(g_lb)

    # --- unopy.Model ---------------------------------------------------------
    model = unopy.Model(unopy.PROBLEM_NONLINEAR, n_vars, x_lb, x_ub,
        unopy.ZERO_BASED_INDEXING)

    # objective
    def _objective(x):
        return problem.objective(x)

    def _objective_gradient(x, gradient):
        gradient[:] = problem.gradient(x)

    model.set_objective(unopy.MINIMIZE, _objective, _objective_gradient)

    # constraints
    if n_con > 0:
        jac_rows, jac_cols = problem.jacobianstructure()
        jac_rows = np.asarray(jac_rows, dtype=np.int32)
        jac_cols = np.asarray(jac_cols, dtype=np.int32)
        n_jac_nnz = len(jac_rows)

        def _constraints(x, constraint_values):
            constraint_values[:] = problem.constraints(x)

        def _jacobian(x, jacobian_values):
            jacobian_values[:] = problem.jacobian(x)

        model.set_constraints(n_con, _constraints, g_lb, g_ub, n_jac_nnz,
            jac_rows, jac_cols, _jacobian)

    # Lagrangian Hessian
    try:
        h_rows, h_cols = problem.hessianstructure()
        h_rows = np.asarray(h_rows, dtype=np.int32)
        h_cols = np.asarray(h_cols, dtype=np.int32)
        n_hess_nnz = len(h_rows)

        def _hessian(x, objective_multiplier, multipliers, hessian_values):
            hessian_values[:] = problem.hessian(x, multipliers, objective_multiplier)

        model.set_lagrangian_hessian(n_hess_nnz, unopy.LOWER_TRIANGLE,
            h_rows, h_cols, _hessian)

        # Match the sign convention used by CyIpoptNLP / IPOPT.
        model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    except (AttributeError, NotImplementedError):
        # No exact Hessian available: Uno will use a quasi-Newton approximation.
        logger.warning("unopy_solver: no Hessian provided; Uno will use a "
				"quasi-Newton approximation.")

    # initial point
    model.set_initial_primal_iterate(np.asarray(problem.x_init(), dtype=np.float64))

    return model


# ---------------------------------------------------------------------------
# UnopySolver – thin wrapper that drives a CyIpoptProblemInterface via unopy
# ---------------------------------------------------------------------------

class UnopySolver:
    """
    Solve an optimisation problem described by a CyIpoptProblemInterface using
    the Uno solver (via its Python interface, unopy).

    This class is the unopy counterpart of CyIpoptSolver.

    Parameters
    ----------
    problem_interface :
        An object implementing the CyIpoptProblemInterface API.
    options : dict, optional
        Key/value pairs forwarded to the Uno solver via
        ``uno_solver.set_option``.
    preset : str, optional
        Uno preset to apply (``"ipopt"`` or ``"filtersqp"``).
        Defaults to ``"ipopt"`` to maximise compatibility with IPOPT-based
        workflows.
    """

    def __init__(self, problem_interface, options=None, preset="ipopt"):
        self._problem = problem_interface
        self._options = options if options is not None else {}
        assert isinstance(self._options, dict)
        self._preset = preset

    def solve(self, x0=None, tee=False):
        """
        Solve the problem.

        Parameters
        ----------
        x0 : array-like, optional
            Initial primal point.  Uses ``problem_interface.x_init()`` when
            *None*.
        tee : bool
            Stream solver output to stdout when *True*.

        Returns
        -------
        primal_solution : np.ndarray
        info : dict
            Dictionary with keys that mirror the ``info`` dict returned by
            cyipopt (``obj_val``, ``mult_g``, ``mult_x_L``, ``mult_x_U``,
            ``status``, ``status_msg``) so that downstream code written for
            cyipopt needs minimal changes.
        """
        model = _build_unopy_model(self._problem)

        # Override the initial point if one was supplied explicitly.
        if x0 is not None:
            model.set_initial_primal_iterate(x0)

        uno_solver = unopy.UnoSolver()
        uno_solver.set_preset(self._preset)
        for option_name, option_value in self._options.items():
            uno_solver.set_option(option_name, option_value)

        with capture_output(sys.stdout if tee else None, capture_fd=True):
            result = uno_solver.optimize(model)

        x = result.primal_solution

        info = {
            "obj_val": result.solution_objective,
            "mult_g": result.constraint_dual_solution,
            "mult_x_L": result.lower_bound_dual_solution,
            "mult_x_U": result.upper_bound_dual_solution,
            "status": str(result.optimization_status),
            "status_msg": str(result.solution_status),
        }

        return x, info


# ---------------------------------------------------------------------------
# PyomoUnopySolver – drives a Pyomo Block directly via unopy
# ---------------------------------------------------------------------------

class PyomoUnopySolver:
    """
    Solve a Pyomo model using the Uno solver (via unopy).

    This class is the unopy counterpart of PyomoCyIpoptSolver.  The Pyomo
    model is first converted to a PyomoNLP (or PyomoNLPWithGreyBoxBlocks),
    then wrapped in a CyIpoptNLP to obtain the standardised callback API, and
    finally translated into a unopy.Model for Uno to solve.
    """

    CONFIG = ConfigBlock("unopy")
    CONFIG.declare(
        "tee",
        ConfigValue(
            default=False, domain=bool, description="Stream solver output to console"
        ),
    )
    CONFIG.declare(
        "load_solutions",
        ConfigValue(
            default=True,
            domain=bool,
            description="Store the final solution into the original Pyomo model",
        ),
    )
    CONFIG.declare(
        "return_nlp",
        ConfigValue(
            default=False,
            domain=bool,
            description="Return the results object and the underlying NLP object.",
        ),
    )
    CONFIG.declare("options", ConfigBlock(implicit=True))
    CONFIG.declare(
        "halt_on_evaluation_error",
        ConfigValue(
            default=None,
            description="Whether to halt if a function or derivative evaluation fails",
        ),
    )
    CONFIG.declare(
        "preset",
        ConfigValue(
            default="ipopt",
            domain=str,
            description=(
                "Uno preset to use.  Either 'ipopt' (default) or 'filtersqp'."
            ),
        ),
    )

    def __init__(self, **kwds):
        self.config = self.CONFIG(kwds)

    def _set_model(self, model):
        self._model = model

    def available(self, exception_flag=False):
        return bool(numpy_available and unopy_available)

    def license_is_valid(self):
        return True

    def version(self):
        try:
            return tuple(int(x) for x in unopy.__version__.split("."))
        except Exception:
            return (0,)

    def solve(self, model, **kwds):
        config = self.config(kwds, preserve_implicit=True)

        if not isinstance(model, Block):
            raise ValueError(
                "PyomoUnopySolver.solve(model): model must be a Pyomo Block"
            )

        # Build the PyomoNLP (handling grey-box blocks if present).
        grey_box_blocks = list(
            model.component_data_objects(egb.ExternalGreyBoxBlock, active=True)
        )
        objectives = list(model.component_data_objects(Objective, active=True))
        n_obj = len(objectives)
        for gbb in grey_box_blocks:
            if gbb.get_external_model().has_objective():
                n_obj += 1
        if n_obj == 0:
            objname = unique_component_name(model, "_obj")
            objective = model.add_component(objname, Objective(expr=0.0))
        try:
            if grey_box_blocks:
                nlp = pyomo_grey_box.PyomoNLPWithGreyBoxBlocks(model)
            else:
                nlp = pyomo_nlp.PyomoNLP(model)
        finally:
            if n_obj == 0:
                model.del_component(objective)

        # Wrap the PyomoNLP in a CyIpoptNLP to get the standardised callback API
        # that _build_unopy_model() expects.
        problem = cyipopt_interface.CyIpoptNLP(
            nlp,
            halt_on_evaluation_error=config.halt_on_evaluation_error,
        )

        ng = len(problem.g_lb())
        nx = len(problem.x_lb())

        # Build and solve the unopy model.
        uno_model = _build_unopy_model(problem)

        uno_solver = unopy.UnoSolver()
        uno_solver.set_preset(config.preset)
        for option_name, option_value in config.options.items():
            uno_solver.set_option(option_name, option_value)

        timer = TicTocTimer()
        try:
            with capture_output(sys.stdout if config.tee else None, capture_fd=True):
                result = uno_solver.optimize(uno_model)
            solverStatus = SolverStatus.ok
        except Exception:
            msg = "Exception encountered during unopy solve:"
            logger.error(msg, exc_info=sys.exc_info())
            solverStatus = SolverStatus.unknown
            raise

        wall_time = timer.toc(None)

        x = result.primal_solution
        mult_g = result.constraint_dual_solution
        mult_x_L = result.lower_bound_dual_solution
        mult_x_U = result.upper_bound_dual_solution
        obj_val = result.solution_objective

        # Load the solution back into the Pyomo model.
        if config.load_solutions:
            nlp.set_primals(x)
            nlp.set_duals(mult_g)
            nlp.load_state_into_pyomo(bound_multipliers=(mult_x_L, mult_x_U))
        else:
            soln = Solution()
            sm = nlp.symbol_map
            soln.variable.update(
                (
                    sm.getSymbol(i),
                    {"Value": j, "uno_zL_out": zl, "uno_zU_out": zu},
                )
                for i, j, zl, zu in zip(
                    nlp.get_pyomo_variables(), x, mult_x_L, mult_x_U
                )
            )
            soln.constraint.update(
                (sm.getSymbol(i), {"Dual": j})
                for i, j in zip(nlp.get_pyomo_constraints(), mult_g)
            )
            model.solutions.add_symbol_map(sm)
            results_obj = SolverResults()
            results_obj._smap_id = id(sm)
            results_obj.solution.insert(soln)

        pyomo_results = SolverResults()

        pyomo_results.problem.name = model.name
        obj = next(model.component_data_objects(Objective, active=True))
        pyomo_results.problem.sense = obj.sense
        if obj.sense == minimize:
            pyomo_results.problem.upper_bound = obj_val
        else:
            pyomo_results.problem.lower_bound = obj_val
        pyomo_results.problem.number_of_objectives = 1
        pyomo_results.problem.number_of_constraints = ng
        pyomo_results.problem.number_of_variables = nx
        pyomo_results.problem.number_of_binary_variables = 0
        pyomo_results.problem.number_of_integer_variables = 0
        pyomo_results.problem.number_of_continuous_variables = nx

        pyomo_results.solver.name = "unopy (Uno)"
        pyomo_results.solver.return_code = str(result.optimization_status)
        pyomo_results.solver.message = str(result.solution_status)
        pyomo_results.solver.wallclock_time = wall_time

        term_cond = _uno_termination_condition(result)
        pyomo_results.solver.termination_condition = term_cond
        pyomo_results.solver.status = TerminationCondition.to_solver_status(term_cond)

        problem.close()

        if config.return_nlp:
            return pyomo_results, nlp

        return pyomo_results

    # Support "with" statements.
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

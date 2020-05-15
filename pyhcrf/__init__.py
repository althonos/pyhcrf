# coding: utf-8

# Copyright (c) 2020, Martin Larralde
# Copyright (c) 2013-2016, Dirko Coetsee

import itertools

import numpy
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import minimize

from ._algorithms import (
    forward,
    _log_likelihood,
    regularize_l1_naive,
    regularize_l1_clipping,
    regularize_l2
)


class _ObjectiveFunction(object):

    def __init__(self, hcrf, X, y):
        self.X = X
        self.y = y
        self.hcrf = hcrf
        self.classes_map = {cls:i for i,cls in enumerate(hcrf.classes_)}

        # store references to the right regularization functions
        self._regularize_l1 = hcrf._regularize_l1
        self._regularize_l2 = regularize_l2

        # store dimensions of the state parameters for faster unstacking
        self._state_parameters_shape = hcrf.state_parameters.shape
        self._state_parameters_count = numpy.prod(hcrf.state_parameters.shape)

        # allocate buffers only once to allow reusing them in several calls;
        # parameters gradient buffer are allocated contiguously to avoid having
        # to stack (which is costly) and unstack (not so much) parameters at
        # each iteration of the loop in __call__: _dstate_parameters and
        # _dtransitions_parameters are unstacked views of _dgradient
        num_state_parameters = numpy.prod(hcrf.state_parameters.shape)
        self._dgradient = numpy.empty(num_state_parameters + hcrf.transition_parameters.shape[0])
        self._dstate_parameters = self._dgradient[:num_state_parameters].reshape(hcrf.state_parameters.shape)
        self._dtransitions_parameters = self._dgradient[num_state_parameters:]

    def __call__(self, parameter_vector, start=None, end=None):
        ll = 0.0
        gradient = numpy.zeros_like(parameter_vector)
        parameters = self._unstack_parameters(parameter_vector)

        for x, ty in itertools.islice(zip(self.X, self.y), start, end):
            # this call updates _dgradient as well since _dstate_parameters and
            # _dtransitions_parameters are memory views of _dgradient
            dll = _log_likelihood(
                x,
                self.classes_map[ty],
                *parameters,
                self.hcrf.transitions,
                self._dstate_parameters,
                self._dtransitions_parameters,
            )
            ll += dll
            gradient += self._dgradient

        # exclude the bias parameters from being regularized
        parameters_without_bias = numpy.array(parameter_vector)
        parameters_without_bias[0] = 0

        # L1/L2 regularization if required
        if self.hcrf.c1 > 0.0:
            ll = self._regularize_l1(ll, gradient, self.hcrf.c1, parameters_without_bias)
        if self.hcrf.c2 > 0.0:
            ll = self._regularize_l2(ll, gradient, self.hcrf.c2, parameters_without_bias)

        # We want to maximize the log-likelihood, so we minizime the opposite
        return -ll, -gradient

    def _unstack_parameters(self, parameter_vector):
        return (
            parameter_vector[:self._state_parameters_count].reshape(self._state_parameters_shape),
            parameter_vector[self._state_parameters_count:],
        )


class HCRF(object):
    """The HCRF model.

    Includes methods for training using LM-BFGS, scoring, and testing, and
    helper methods for loading and saving parameter values to and from file.

    Attributes:
        classes_ (list of str, optional): The list of classes known by the
            model. It contains all class labels given when the model was fit,
            or `None` is the model has not been fitted yet.

    """

    _L1_STRATEGY = {
        "naive": regularize_l1_naive,
        "clipping": regularize_l1_clipping,
        # TODO: implement penalty
        "penalty": regularize_l1_clipping,
    }

    def __init__(
        self,
        num_states=2,
        c1=0.15,
        c2=0.15,
        transitions=None,
        state_parameter_noise=0.001,
        transition_parameter_noise=0.001,
        optimizer_kwargs=None,
        sgd_stepsize=None,
        l1_strategy="clipping",
    ):
        """Instantiate a HCRF with hidden units of cardinality ``num_states``.

        Arguments:
            num_states (int): The number of hidden states to consider.
            c1 (float): The value of the weight parameter to use for L1
                regularization (LASSO).
            c2 (float): The value of the weight parameter to use for L2
                regularization (Ridge Regression).
            transitions (2D array, optional): An array of transitions to use.
                Each row of the array should be of the form :math:`(c, s0, s1)`
                where :math:`c` is a class index, :math:`s0` and :math:`s1`
                the indices of two states.
            state_parameter_noise (float): The noise to apply to initial state
                parameters when training the model.
            transition_parameter_noise (float): The noise to apply to initial
                transition parameters when training the model.
            optimizer_kwargs (dict, optional): A dictionary with additional
                parameters to pass to the optimizer. See the documentation of
                :py:`scipy.optimize.minimize`.
            sgd_stepsize (float, optional): The SGD step size to use when
                building the initial parameters vector before minimization.
            l1_strategy (str): The L1 regularization strategy to use to
                approximate the L1 norm gradient. Should be one of ``naive``,
                ``clipping`` (default), or ``penalty``. See [Tsuruoka 09].

        References:
            .. [Tsuruoka 09]: `Yoshimasa Tsuruoka, Jun’ichi Tsujii, and Sophia
               Ananiadou, ‘Stochastic Gradient Descent Training for L1-Regularized
               Log-Linear Models with Cumulative Penalty’, in Proceedings of the
               Joint Conference of the 47th Annual Meeting of the ACL and the
               4th International Joint Conference on Natural Language Processing
               of the AFNLP (ACL-IJCNLP 2009, Suntec, Singapore: Association
               for Computational Linguistics, 2009), 477–485,
               <https://www.aclweb.org/anthology/P09-1054>`_.

        """
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive, not {}".format(num_states))

        # Make sure to store transitions in a numpy array of `int64`,
        # otherwise could cause error when calling `_algorithms` functions.
        if transitions is not None:
            self.transitions = numpy.array(transitions, dtype="int64")
        else:
            self.transitions = transitions

        # use the appropriate L1 regularization function depending on the
        # chosen strategy
        if l1_strategy in self._L1_STRATEGY:
            self._regularize_l1 = self._L1_STRATEGY[l1_strategy]
        else:
            raise ValueError("unknown value for l1_strategy: {}".format(l1_strategy))

        # Attributes provided for compatibility with sklearn_crfsuite.CRF
        self.classes_ = None
        self.attributes_ = None
        self.algorithm = "lbsgf"
        self.c1 = c1
        self.c2 = c2

        # Other parameters
        self.l1_strategy = l1_strategy
        self.num_states = num_states
        self.parameters = None
        self.state_parameters = None
        self.transition_parameters = None
        self.state_parameter_noise = state_parameter_noise
        self.transition_parameter_noise = transition_parameter_noise
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.sgd_stepsize = sgd_stepsize

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (array, shape :math:`(n_samples, n_stimesteps, n_features)`):
                List of list of list of ints. Each list of ints represent an
                observation, and is expected to be a one-hot encoded list of
                features. :math:`n_timesteps` can vary between different
                samples.
            y (array, shape :math:`(n_samples)`):
                Target vector relative to X containing labels for each samples.
                Labels can be any hashable Python object.

        """
        self.classes_ = sorted(set(y))
        num_classes = len(self.classes_)
        if self.transitions is None:
            self.transitions = self._create_default_transitions(
                num_classes, self.num_states
            )

        # Allocate buffers for the parameters, using views to avoid
        # stacking / unstacking
        _, num_features = X[0].shape
        num_transitions, _ = self.transitions.shape
        state_parameters_shape = (num_features, self.num_states, num_classes)
        state_parameters_count = numpy.prod(state_parameters_shape)
        self.parameters = numpy.empty( state_parameters_count + num_transitions )
        self.state_parameters = self.parameters[:state_parameters_count].reshape(state_parameters_shape)
        self.transition_parameters = self.parameters[state_parameters_count:]

        # Initialize parameters with random values
        s = numpy.random.standard_normal(self.state_parameters.shape)
        numpy.copyto(self.state_parameters, s * self.state_parameter_noise)
        t = numpy.random.standard_normal(self.transition_parameters.shape)
        numpy.copyto(self.transition_parameters, t * self.transition_parameter_noise)

        # Initialize the objective function using the training parameters
        objective_function = _ObjectiveFunction(self, X, y)

        # If the stochastic gradient stepsize is defined, do 1 epoch of SGD to initialize the parameters.
        if self.sgd_stepsize is not None:
            total_nll = 0.0
            for i in range(len(y)):
                nll, ngradient = objective_function(self.parameters, i, i + 1)
                total_nll += nll
                initial_parameter_vector -= ngradient * self.sgd_stepsize

        # Run the optimizer to maximize the log-likelihood
        self._optimizer_result = minimize(
            objective_function,
            self.parameters,
            method="L-BFGS-B",
            jac=True,
            **self.optimizer_kwargs
        )

        # Check the optimizer converged and update the optimal parameters
        if not self._optimizer_result.success:
            raise RuntimeError("optimize did not converge: {}".format(self._optimizer_result.message))
        numpy.copyto(self.parameters, self._optimizer_result.x)

        return self

    def predict(self, X):
        """Predict the class for each sample in ``X``.

        Arguments:
            X (array, shape :math:`(n_samples, n_timesteps, n_features)`): The
                samples to predict labels for. **Samples must have the same
                number of features as the samples used for training**.

        Returns:
            array, shape :math:`(n_samples)`: An array containing the class
            label with the highest probability for each sample of ``X``. Use
            :py:`HCRF.predict_marginals` to access all class probabilities

        """
        preds = self.predict_marginals(X)
        return numpy.array([max(pred, key=pred.__getitem__) for pred in preds])

    def predict_marginals(self, X):
        """Estimate all class probabilities for each sample in ``X``.

        The returned estimates for all classes are ordered by the
        label of classes.

        Arguments:
            X (array, shape :math:`(n_samples, n_timesteps, n_features)`): The
                samples to predict labels for. **Samples must have the same
                number of features as the samples used for training**.

        Returns:
            array, shape :math:`(n_samples)`: An array containing dictionaries
            mapping class labels to the prediction probability for each sample
            of ``X``.

        """
        y = []
        for x in X:
            n_time_steps, n_features = x.shape
            _, n_states, n_classes = self.state_parameters.shape
            x_dot_parameters = x.dot(
                self.state_parameters.reshape(n_features, -1)
            ).reshape((n_time_steps, n_states, n_classes))
            forward_table = forward(
                x_dot_parameters,
                self.state_parameters,
                self.transition_parameters,
                self.transitions,
            )
            # TODO: normalize by subtracting log-sum to avoid overflow
            preds = numpy.exp(forward_table[-1, -1, :])
            y.append(dict(zip(self.classes_, preds / preds.sum())))
        return numpy.array(y)

    @staticmethod
    def _create_default_transitions(num_classes, num_states):
        # 0    o>
        # 1    o>\\\
        # 2   /o>/||
        # 3  |/o>//
        # 4  \\o>/
        transitions = []
        for c in range(num_classes):  # The zeroth state
            transitions.append([c, 0, 0])
        for state in range(0, num_states - 1):  # Subsequent states
            for c in range(num_classes):
                transitions.append([c, state, state + 1])  # To the next state
                transitions.append([c, state + 1, state + 1])  # Stays in same state
                if state > 0:
                    transitions.append([c, 0, state + 1])  # From the start state
                if state < num_states - 1:
                    transitions.append(
                        [c, state + 1, num_states - 1]
                    )  # To the end state
        transitions = numpy.array(transitions, dtype="int64")
        return transitions

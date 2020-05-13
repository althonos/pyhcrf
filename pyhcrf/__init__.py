# coding: utf-8

# Copyright (c) 2020, Martin Larralde
# Copyright (c) 2013-2016, Dirko Coetsee
#
# This file is part of pyhcrf.
#
# pyhcrf is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pyhcrf is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with pyhcrf.  If not, see <https://www.gnu.org/licenses/>.

import numpy
from scipy.optimize.lbfgsb import fmin_l_bfgs_b

from ._algorithms import forward_backward, log_likelihood


class HCRF(object):
    """The HCRF model.

    Includes methods for training using LM-BFGS, scoring, and testing, and
    helper methods for loading and saving parameter values to and from file.

    Attributes:
        classes_ (list of str, optional): The list of classes known by the
            model. It contains all class labels given when the model was fit,
            or `None` is the model has not been fitted yet.
        attributes_ (list of str, optional): The list of features known by the
            model. It contains all features given when the model was fit,
            or `None` if the model has not been fitted yet.

    """

    def __init__(
        self,
        num_states=2,
        l2_regularization=1.0,
        transitions=None,
        state_parameter_noise=0.001,
        transition_parameter_noise=0.001,
        optimizer_kwargs=None,
        sgd_stepsize=None,
        random_seed=0,
    ):
        """Instantiate a HCRF with hidden units of cardinality ``num_states``.
        """
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive, not {}".format(num_states))

        # Make sure to store transitions in a numpy array of `int64`,
        # otherwise could cause error when calling `_algorithms` functions.
        if transitions is not None:
            self.transitions = numpy.array(transitions, dtype="int64")
        else:
            self.transitions = transitions

        # Attributes provided for compatibility with sklearn_crfsuite.CRF
        self.classes_ = None
        self.attributes_ = None
        self.algorithm = "lbsgf"

        # Other parameters
        self.l2_regularization = l2_regularization
        self.num_states = num_states
        self.state_parameters = None
        self.transition_parameters = None
        self.state_parameter_noise = state_parameter_noise
        self.transition_parameter_noise = transition_parameter_noise
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.sgd_stepsize = sgd_stepsize
        self._random_seed = random_seed

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : List of list of ints. Each list of ints represent a training example.
            Each int in that list must be the index of a one-hot encoded feature.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = list(set(y))
        num_classes = len(self.classes_)
        classes_map = {cls:i for i,cls in enumerate(self.classes_)}
        if self.transitions is None:
            self.transitions = self._create_default_transitions(
                num_classes, self.num_states
            )

        # Initialise the parameters
        _, num_features = X[0].shape
        num_transitions, _ = self.transitions.shape
        numpy.random.seed(self._random_seed)
        if self.state_parameters is None:
            self.state_parameters = (
                numpy.random.standard_normal(
                    (num_features, self.num_states, num_classes)
                )
                * self.state_parameter_noise
            )
        if self.transition_parameters is None:
            self.transition_parameters = (
                numpy.random.standard_normal((num_transitions))
                * self.transition_parameter_noise
            )

        initial_parameter_vector = self._stack_parameters(
            self.state_parameters, self.transition_parameters
        )
        function_evaluations = [0]

        def objective_function(parameter_vector):
            ll = 0.0
            gradient = numpy.zeros_like(parameter_vector)
            state_parameters, transition_parameters = self._unstack_parameters(
                parameter_vector
            )
            for x, ty in zip(X, y):
                y_index = classes_map[ty]
                dll, dgradient_state, dgradient_transition = log_likelihood(
                    x,
                    y_index,
                    state_parameters,
                    transition_parameters,
                    self.transitions,
                )
                dgradient = self._stack_parameters(
                    dgradient_state, dgradient_transition
                )
                ll += dll
                gradient += dgradient

            # exclude the bias parameters from being regularized
            parameters_without_bias = numpy.array(parameter_vector)
            parameters_without_bias[0] = 0
            ll -= self.l2_regularization * numpy.dot(
                parameters_without_bias.T, parameters_without_bias
            )
            gradient = (
                gradient.flatten()
                - 2.0 * self.l2_regularization * parameters_without_bias
            )

            function_evaluations[0] += 1
            return -ll, -gradient

        # If the stochastic gradient stepsize is defined, do 1 epoch of SGD to initialize the parameters.
        if self.sgd_stepsize is not None:
            total_nll = 0.0
            for i in range(len(y)):
                nll, ngradient = objective_function(initial_parameter_vector, i, i + 1)
                total_nll += nll
                initial_parameter_vector -= ngradient * self.sgd_stepsize

        self._optimizer_result = fmin_l_bfgs_b(
            objective_function, initial_parameter_vector, **self.optimizer_kwargs
        )
        self.state_parameters, self.transition_parameters = self._unstack_parameters(
            self._optimizer_result[0]
        )
        return self

    def predict(self, X):
        """Predict the class for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : List of list of ints, one list of ints for each training example.

        Returns
        -------
        y : iterable of shape = [n_samples]
            The predicted classes.
        """
        return [
            self.classes_[prediction.argmax()]
            for prediction in self.predict_marginals(X)
        ]

    def predict_marginals(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        y = []
        for x in X:
            n_time_steps, n_features = x.shape
            _, n_states, n_classes = self.state_parameters.shape
            x_dot_parameters = x.dot(
                self.state_parameters.reshape(n_features, -1)
            ).reshape((n_time_steps, n_states, n_classes))
            forward_table, _, _ = forward_backward(
                x_dot_parameters,
                self.state_parameters,
                self.transition_parameters,
                self.transitions,
            )
            # TODO: normalize by subtracting log-sum to avoid overflow
            y.append(
                numpy.exp(forward_table[-1, -1, :])
                / sum(numpy.exp(forward_table[-1, -1, :]))
            )
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

    @staticmethod
    def _stack_parameters(state_parameters, transition_parameters):
        return numpy.concatenate((state_parameters.flatten(), transition_parameters))

    def _unstack_parameters(self, parameter_vector):
        state_parameter_shape = self.state_parameters.shape
        num_state_parameters = numpy.prod(state_parameter_shape)
        return (
            parameter_vector[:num_state_parameters].reshape(state_parameter_shape),
            parameter_vector[num_state_parameters:],
        )

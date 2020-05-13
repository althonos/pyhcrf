# coding: utf-8
# cython: language_level=3, linetrace=True

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


from libc.stdint cimport int32_t, uint32_t, int64_t
from numpy cimport ndarray
from numpy.math cimport INFINITY as inf

import numpy


cdef extern from "<math.h>":
    double exp(double x)

cdef extern from "logaddexp.h":
    double logaddexp(double x, double y)


def forward_backward(
    ndarray[double, ndim=3] x_dot_parameters,
    ndarray[double, ndim=3] state_parameters,
    ndarray[double, ndim=1] transition_parameters,
    ndarray[int64_t, ndim=2] transitions,
):
    cdef uint32_t n_time_steps = x_dot_parameters.shape[0]
    cdef uint32_t n_states = state_parameters.shape[1]
    cdef uint32_t n_classes = state_parameters.shape[2]

    cdef uint32_t n_transitions = transitions.shape[0]

    # Add extra 1 time step for start state
    cdef ndarray[double, ndim=3] forward_table = numpy.full((n_time_steps + 1, n_states, n_classes), fill_value=-inf, dtype='float64')
    cdef ndarray[double, ndim=4] forward_transition_table = numpy.full((n_time_steps + 1, n_states, n_states, n_classes), fill_value=-inf, dtype='float64')
    forward_table[0, 0, :] = 0.0

    cdef ndarray[double, ndim=3] backward_table = numpy.full((n_time_steps + 1, n_states, n_classes), fill_value=-inf, dtype='float64')
    backward_table[-1, -1, :] = 0.0

    cdef uint32_t class_number, s0, s1
    cdef int32_t t
    cdef double edge_potential

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            class_number = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = forward_table[t - 1, s0, class_number] + transition_parameters[transition]
            forward_table[t, s1, class_number] = logaddexp(forward_table[t, s1, class_number],
                                                                 edge_potential + x_dot_parameters[t - 1, s1, class_number])
            forward_transition_table[t, s0, s1, class_number] = logaddexp(forward_transition_table[t, s0, s1, class_number],
                                                                                edge_potential +
                                                                                x_dot_parameters[t - 1, s1, class_number])

    for t in range(n_time_steps - 1, -1, -1):
        for transition in range(n_transitions):
            class_number = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = (backward_table[t + 1, s1, class_number] + x_dot_parameters[t, s1, class_number])
            backward_table[t, s0, class_number] = logaddexp(backward_table[t, s0, class_number],
                                                                  edge_potential + transition_parameters[transition])

    return forward_table, forward_transition_table, backward_table

def dummy():
    pass

def log_likelihood(
    x,
    int64_t cy,
    ndarray[double, ndim=3] state_parameters,
    ndarray[double, ndim=1] transition_parameters,
    ndarray[int64_t, ndim=2] transitions,
):
    cdef uint32_t n_time_steps = x.shape[0]
    cdef uint32_t n_features = x.shape[1]
    cdef uint32_t n_states = state_parameters.shape[1]
    cdef uint32_t n_classes = state_parameters.shape[2]
    cdef ndarray[double, ndim=3] x_dot_parameters = x.dot(state_parameters.reshape(n_features, -1)).reshape((n_time_steps, n_states, n_classes))

    cdef ndarray[double, ndim=3] forward_table
    cdef ndarray[double, ndim=4] forward_transition_table
    cdef ndarray[double, ndim=3] backward_table

    (forward_table,
     forward_transition_table,
     backward_table) = forward_backward(x_dot_parameters,
                                        state_parameters,
                                        transition_parameters,
                                        transitions)
    n_time_steps = forward_table.shape[0] - 1
    cdef uint32_t n_transitions = transitions.shape[0]
    cdef ndarray[double, ndim=3] dstate_parameters = numpy.zeros_like(state_parameters, dtype='float64')
    cdef ndarray[double, ndim=1] dtransition_parameters = numpy.zeros_like(transition_parameters, dtype='float64')

    cdef ndarray[double, ndim=1] class_Z = numpy.empty((n_classes,))
    cdef double Z = -inf
    cdef uint32_t c
    for c in range(n_classes):
        class_Z[c] = forward_table[-1, -1, c]
        Z = logaddexp(Z, forward_table[-1, -1, c])

    cdef uint32_t t, state, transition, s0, s1
    cdef double alphabeta
    for t in range(1, n_time_steps + 1):
        for state in range(n_states):
            for c in range(n_classes):
                alphabeta = forward_table[t, state, c] + backward_table[t, state, c]
                if c == cy:
                    dstate_parameters[:, state, c] += ((exp(alphabeta - class_Z[c]) -
                                                        exp(alphabeta - Z)) * x[t - 1, :])
                else:
                    dstate_parameters[:, state, c] -= exp(alphabeta - Z) * x[t - 1, :]

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            c = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            alphabeta = forward_transition_table[t, s0, s1, c] + backward_table[t, s1, c]
            if c == cy:
                dtransition_parameters[transition] += (exp(alphabeta - class_Z[c]) - exp(alphabeta - Z))
            else:
                dtransition_parameters[transition] -= exp(alphabeta - Z)

    return class_Z[cy] - Z, dstate_parameters, dtransition_parameters

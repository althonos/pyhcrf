# coding: utf-8
# cython: language_level=3, linetrace=True, boundscheck=False, wraparound=True

# Copyright (c) 2020, Martin Larralde
# Copyright (c) 2013-2016, Dirko Coetsee

from libc.stdint cimport int32_t, uint32_t, int64_t
from libc.stddef cimport size_t
from numpy cimport ndarray, float64_t
from numpy.math cimport INFINITY as inf

import numpy


cdef extern from "<math.h>":
    double exp(double x) nogil

cdef extern from "logaddexp.h":
    double logaddexp(double x, double y) nogil


cpdef forward_backward(
    ndarray[float64_t, ndim=3] x_dot_parameters,
    ndarray[float64_t, ndim=3] state_parameters,
    ndarray[float64_t, ndim=1] transition_parameters,
    ndarray[int64_t, ndim=2] transitions,
):
    cdef uint32_t class_number, s0, s1
    cdef float64_t edge_potential
    cdef size_t n_classes, n_states, n_time_steps, n_transitions, t
    cdef ndarray[float64_t, ndim=3] forward_table, backward_table
    cdef ndarray[float64_t, ndim=4] forward_transition_table

    # Extract dimensions of the input tables
    n_time_steps = x_dot_parameters.shape[0]
    n_states = state_parameters.shape[1]
    n_classes = state_parameters.shape[2]
    n_transitions = transitions.shape[0]

    # Add extra 1 time step for start state
    forward_table = numpy.full(
        (n_time_steps + 1, n_states, n_classes),
        fill_value=-inf,
        dtype='float64'
    )
    forward_transition_table = numpy.full(
        (n_time_steps + 1, n_states, n_states, n_classes),
        fill_value=-inf,
        dtype='float64'
    )
    backward_table = numpy.full(
        (n_time_steps + 1, n_states, n_classes),
        fill_value=-inf,
        dtype='float64'
    )
    forward_table[0, 0, :] = 0.0
    backward_table[n_time_steps, n_states-1, :] = 0.0

    with nogil:
        for t in range(1, n_time_steps + 1):
            for transition in range(n_transitions):
                class_number = transitions[transition, 0]
                s0 = transitions[transition, 1]
                s1 = transitions[transition, 2]
                edge_potential = forward_table[t - 1, s0, class_number] + transition_parameters[transition]
                forward_table[t, s1, class_number] = logaddexp(
                    forward_table[t, s1, class_number],
                    edge_potential + x_dot_parameters[t - 1, s1, class_number]
                )
                forward_transition_table[t, s0, s1, class_number] = logaddexp(
                    forward_transition_table[t, s0, s1, class_number],
                    edge_potential + x_dot_parameters[t - 1, s1, class_number]
                )
        for t in range(n_time_steps - 1, -1, -1):
            for transition in range(n_transitions):
                class_number = transitions[transition, 0]
                s0 = transitions[transition, 1]
                s1 = transitions[transition, 2]
                edge_potential = backward_table[t + 1, s1, class_number] + x_dot_parameters[t, s1, class_number]
                backward_table[t, s0, class_number] = logaddexp(
                    backward_table[t, s0, class_number],
                    edge_potential + transition_parameters[transition]
                )

    return forward_table, forward_transition_table, backward_table


cpdef log_likelihood(
    ndarray[float64_t, ndim=2] x,
    size_t cy,
    ndarray[float64_t, ndim=3] state_parameters,
    ndarray[float64_t, ndim=1] transition_parameters,
    ndarray[int64_t, ndim=2] transitions,
):
    #
    cdef float64_t alphabeta, weight, Z
    cdef int64_t s0, s1
    cdef size_t c, feat, t, state, transition
    cdef size_t n_time_steps, n_features, n_states, n_classes, n_transitions
    cdef ndarray[float64_t, ndim=1] dtransition_parameters, class_Z
    cdef ndarray[float64_t, ndim=3] dstate_parameters
    cdef ndarray[float64_t, ndim=3] backward_table, forward_table, x_dot_parameters
    cdef ndarray[float64_t, ndim=4] forward_transition_table

    # Extract dimensions of input arrays
    n_time_steps = x.shape[0]
    n_features = x.shape[1]
    n_states = state_parameters.shape[1]
    n_classes = state_parameters.shape[2]
    n_transitions = transitions.shape[0]

    # Initialize temporary arrays
    dstate_parameters = numpy.zeros_like(state_parameters, dtype='float64')
    dtransition_parameters = numpy.zeros_like(transition_parameters, dtype='float64')
    class_Z = numpy.empty((n_classes,))

    # Compute (x @ state_parameters) before the loop
    x_dot_parameters = (
        x.dot(state_parameters.reshape(n_features, -1))
          .reshape((n_time_steps, n_states, n_classes))
    )

    # Compute the state and transition tables from the given parameters
    forward_table, forward_transition_table, backward_table = forward_backward(
        x_dot_parameters,
        state_parameters,
        transition_parameters,
        transitions
    )

    with nogil:
        # compute Z by rewinding the forward table for all classes
        Z = -inf
        for c in range(n_classes):
            class_Z[c] = forward_table[-1, -1, c]
            Z = logaddexp(Z, forward_table[-1, -1, c])

        # compute all state parameters
        for t in range(1, n_time_steps + 1):
            for state in range(n_states):
                for c in range(n_classes):
                    alphabeta = forward_table[t, state, c] + backward_table[t, state, c]
                    weight = exp(alphabeta - class_Z[c]) * (c == cy) - exp(alphabeta - Z)
                    for feat in range(n_features):
                        dstate_parameters[feat, state, c] += weight * x[t - 1, feat]

        # compute all transition parameters
        for t in range(1, n_time_steps + 1):
            for transition in range(n_transitions):
                c = transitions[transition, 0]
                s0 = transitions[transition, 1]
                s1 = transitions[transition, 2]
                alphabeta = forward_transition_table[t, s0, s1, c] + backward_table[t, s1, c]
                weight = exp(alphabeta - class_Z[c]) * (c == cy) - exp(alphabeta - Z)
                dtransition_parameters[transition] += weight

    return class_Z[cy] - Z, dstate_parameters, dtransition_parameters

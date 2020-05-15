# coding: utf-8
# cython: language_level=3, linetrace=True, boundscheck=False, wraparound=False

# Copyright (c) 2020, Martin Larralde
# Copyright (c) 2013-2016, Dirko Coetsee

from libc.stdint cimport uint8_t, int32_t, uint32_t, int64_t
from libc.stddef cimport size_t
from libc.string cimport memset
from libc.stdio cimport printf
from numpy cimport ndarray, float64_t
from numpy.math cimport INFINITY as inf

cimport numpy
import numpy

cdef extern from "<math.h>":
    double exp(double x) nogil

cdef extern from "logaddexp.h":
    double logaddexp(double x, double y) nogil


# ----------------------------------------------------------------------------


cpdef ndarray sign(ndarray x):
    return (0.0 < x[:]).astype(x.dtype) - (x[:] < 0.0)


cpdef float64_t regularize_l1_naive(
    float64_t ll,
    ndarray[float64_t, ndim=1] gradient,
    float64_t c1,
    ndarray[float64_t, ndim=1] parameters,
):
    gradient -= c1 * sign(parameters)
    ll -= c1 * numpy.abs(parameters).sum()
    return ll

cpdef float64_t regularize_l1_clipping(
    float64_t ll,
    ndarray[float64_t, ndim=1] gradient,
    float64_t c1,
    ndarray[float64_t, ndim=1] parameters,
):
    cdef ndarray[uint8_t, ndim=1] mask = (gradient < c1) & (gradient > -c1)
    ll -= c1 * numpy.abs(parameters).sum()
    gradient -= c1 * sign(parameters)
    gradient *= ~mask
    return ll


cpdef float64_t regularize_l2(
    float64_t ll,
    ndarray[float64_t, ndim=1] gradient,
    float64_t c2,
    ndarray[float64_t, ndim=1] parameters,
):
    ll -= c2 * parameters.dot(parameters.T)
    gradient -= 2.0 * c2 * parameters
    return ll


# ----------------------------------------------------------------------------


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
        # Compute forward transitions
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
        # Compute backwards transitions
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

cpdef ndarray[float64_t, ndim=3] forward(
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
    forward_table[0, 0, :] = 0.0

    with nogil:
        # Compute forward transitions
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

    return forward_table


# ----------------------------------------------------------------------------


def log_likelihood(
    x,
    cy,
    state_parameters,
    transition_parameters,
    transitions,
    dstate_parameters=None,
    dtransition_parameters=None,
    class_Z=None,
):
    if dstate_parameters is None:
        dstate_parameters = numpy.empty_like(state_parameters, dtype="float64")
    if dtransition_parameters is None:
        dtransition_parameters = numpy.empty_like(transition_parameters, dtype="float64")
    dll = _log_likelihood(
        x,
        cy,
        state_parameters,
        transition_parameters,
        transitions,
        dstate_parameters,
        dtransition_parameters,
    )
    return dll, dstate_parameters, dtransition_parameters


cpdef float64_t _log_likelihood(
    ndarray[float64_t, ndim=2] x,
    size_t cy,
    ndarray[float64_t, ndim=3] state_parameters,
    ndarray[float64_t, ndim=1] transition_parameters,
    ndarray[int64_t, ndim=2] transitions,
    float64_t[:,:,:] dstate_parameters,
    float64_t[:] dtransition_parameters,
):
    #
    cdef float64_t alphabeta, weight, Z
    cdef int64_t s0, s1
    cdef size_t c, feat, t, state, transition
    cdef size_t n_time_steps, n_features, n_states, n_classes, n_transitions
    cdef ndarray[float64_t, ndim=3] backward_table, forward_table, x_dot_parameters
    cdef ndarray[float64_t, ndim=4] forward_transition_table

    # Extract dimensions of input arrays
    n_time_steps = x.shape[0]
    n_features = x.shape[1]
    n_states = state_parameters.shape[1]
    n_classes = state_parameters.shape[2]
    n_transitions = transitions.shape[0]

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
        # reset parameter gradients buffers
        dstate_parameters[:,:,:] = 0.0
        dtransition_parameters[:] = 0.0

        # compute Z by rewinding the forward table for all classes
        Z = -inf
        for c in range(n_classes):
            # class_Z[c] = forward_table[n_time_steps, n_states-1, c]
            Z = logaddexp(Z, forward_table[n_time_steps, n_states-1, c])

        # compute all state parameter gradients
        for t in range(1, n_time_steps + 1):
            for state in range(n_states):
                for c in range(n_classes):
                    alphabeta = forward_table[t, state, c] + backward_table[t, state, c]
                    weight = exp(alphabeta - forward_table[n_time_steps, n_states-1, c]) * (c == cy) - exp(alphabeta - Z)
                    for feat in range(n_features):
                        dstate_parameters[feat, state, c] += weight * x[t - 1, feat]

        # compute all transition parameter gradients
        for t in range(1, n_time_steps + 1):
            for transition in range(n_transitions):
                c = transitions[transition, 0]
                s0 = transitions[transition, 1]
                s1 = transitions[transition, 2]
                alphabeta = forward_transition_table[t, s0, s1, c] + backward_table[t, s1, c]
                weight = exp(alphabeta - forward_table[n_time_steps, n_states-1, c]) * (c == cy) - exp(alphabeta - Z)
                dtransition_parameters[transition] += weight

        return forward_table[n_time_steps, n_states-1, cy] - Z

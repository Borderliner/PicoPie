# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Compiled bulk transfer between NumPy and the native PicoGK runtime.

The C ABI exposes only per-element getters/setters. Looping over them from
Python (one ctypes call each) is slow. Here we take the raw address of a native
function (obtained from ctypes) and call it through a typed function pointer in
a ``nogil`` loop, writing straight into a contiguous NumPy buffer -- so the
whole transfer costs one Python call plus a tight compiled loop.

All functions take ``fn`` = the native function's address (an int), the library
instance and object handles, and operate on C-contiguous NumPy arrays whose
memory layout matches the packed C structs (PKVector3 = 3 float32, PKTriangle =
3 int32). ``bool``-returning natives are read as a 1-byte ``unsigned char`` to
match the C ABI exactly.
"""

import numpy as np
from libc.math cimport NAN

ctypedef unsigned long long u64

# Native function-pointer signatures (see API/PicoGK.h).
ctypedef void          (*getvec_t)(u64, u64, int, float*)   noexcept nogil
ctypedef void          (*gettri_t)(u64, u64, int, int*)     noexcept nogil
ctypedef int           (*addvec_t)(u64, u64, float*)        noexcept nogil
ctypedef int           (*addtri_t)(u64, u64, int*)          noexcept nogil
ctypedef void          (*setscl_t)(u64, u64, float*, float) noexcept nogil
ctypedef unsigned char (*getscl_t)(u64, u64, float*, float*) noexcept nogil
ctypedef void          (*setvec_t)(u64, u64, float*, float*) noexcept nogil
ctypedef unsigned char (*getvecf_t)(u64, u64, float*, float*) noexcept nogil


# --- Mesh: read --------------------------------------------------------------
def read_vertices(u64 fn, u64 inst, u64 h, int n):
    out = np.empty((n, 3), dtype=np.float32)
    if n == 0:
        return out
    cdef float[:, ::1] mv = out
    cdef getvec_t f = <getvec_t><void*>(<size_t>fn)
    cdef int i
    with nogil:
        for i in range(n):
            f(inst, h, i, &mv[i, 0])
    return out


def read_triangles(u64 fn, u64 inst, u64 h, int n):
    out = np.empty((n, 3), dtype=np.int32)
    if n == 0:
        return out
    cdef int[:, ::1] mv = out
    cdef gettri_t f = <gettri_t><void*>(<size_t>fn)
    cdef int i
    with nogil:
        for i in range(n):
            f(inst, h, i, &mv[i, 0])
    return out


# --- Mesh: build -------------------------------------------------------------
def add_vertices(u64 fn, u64 inst, u64 h, float[:, ::1] verts):
    cdef addvec_t f = <addvec_t><void*>(<size_t>fn)
    cdef Py_ssize_t i, n = verts.shape[0]
    with nogil:
        for i in range(n):
            f(inst, h, &verts[i, 0])


def add_triangles(u64 fn, u64 inst, u64 h, int[:, ::1] tris):
    cdef addtri_t f = <addtri_t><void*>(<size_t>fn)
    cdef Py_ssize_t i, n = tris.shape[0]
    with nogil:
        for i in range(n):
            f(inst, h, &tris[i, 0])


# --- ScalarField: bulk get/set ----------------------------------------------
def scalar_set_many(u64 fn, u64 inst, u64 h, float[:, ::1] pos, float[::1] vals):
    cdef setscl_t f = <setscl_t><void*>(<size_t>fn)
    cdef Py_ssize_t i, n = pos.shape[0]
    with nogil:
        for i in range(n):
            f(inst, h, &pos[i, 0], vals[i])


def scalar_get_many(u64 fn, u64 inst, u64 h, float[:, ::1] pos):
    cdef Py_ssize_t n = pos.shape[0]
    out = np.empty(n, dtype=np.float32)
    found = np.zeros(n, dtype=np.uint8)
    cdef float[::1] mo = out
    cdef unsigned char[::1] mf = found
    cdef getscl_t f = <getscl_t><void*>(<size_t>fn)
    cdef Py_ssize_t i
    cdef float tmp
    with nogil:
        for i in range(n):
            if f(inst, h, &pos[i, 0], &tmp) != 0:
                mo[i] = tmp
                mf[i] = 1
            else:
                mo[i] = NAN
    return out, found.view(bool)


# --- VectorField: bulk get/set ----------------------------------------------
def vector_set_many(u64 fn, u64 inst, u64 h, float[:, ::1] pos, float[:, ::1] vals):
    cdef setvec_t f = <setvec_t><void*>(<size_t>fn)
    cdef Py_ssize_t i, n = pos.shape[0]
    with nogil:
        for i in range(n):
            f(inst, h, &pos[i, 0], &vals[i, 0])


def vector_get_many(u64 fn, u64 inst, u64 h, float[:, ::1] pos):
    cdef Py_ssize_t n = pos.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    found = np.zeros(n, dtype=np.uint8)
    cdef float[:, ::1] mo = out
    cdef unsigned char[::1] mf = found
    cdef getvecf_t f = <getvecf_t><void*>(<size_t>fn)
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            if f(inst, h, &pos[i, 0], &mo[i, 0]) != 0:
                mf[i] = 1
    fb = found.view(bool)
    out[~fb] = NAN
    return out, fb

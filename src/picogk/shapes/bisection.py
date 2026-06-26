"""Bisection root finder (port of ShapeKernel ``Bisection``).

Finds an input in ``[min_input, max_input]`` whose function value equals a
target, to an absolute tolerance ``epsilon``. The bracket must straddle the
target (opposite signs of ``func - target`` at the ends).
"""

from __future__ import annotations

from collections.abc import Callable


class BisectionError(Exception):
    """Raised when the bracket is invalid or no solution is reached in time."""


class Bisection:
    """Approximate ``func(x) == target`` by interval bisection."""

    def __init__(self, func: Callable[[float], float], min_input: float,
                 max_input: float, target: float, epsilon: float = 0.01,
                 max_iterations: int = 500):
        self._func = func
        self._min_input = float(min_input)
        self._max_input = float(max_input)
        self._target = float(target)
        self._epsilon = float(epsilon)
        self._max_iterations = int(max_iterations)
        self._iterations = 0
        self._remaining_diff = 0.0
        self._best_guess = 0.0

    def _residual(self, x: float) -> float:
        return self._func(x) - self._target

    def solve(self) -> float:
        """Return an input that yields the target value (within ``epsilon``)."""
        lo = self._min_input
        hi = self._max_input
        if self._residual(lo) * self._residual(hi) >= 0:
            raise BisectionError("no valid limits (bracket must straddle the target)")

        mid = lo
        self._remaining_diff = hi - lo
        while self._remaining_diff >= self._epsilon:
            mid = 0.5 * (lo + hi)
            r_mid = self._residual(mid)
            if r_mid == 0.0:
                break
            if r_mid * self._residual(lo) < 0:
                hi = mid
            else:
                lo = mid

            self._remaining_diff = hi - lo
            self._iterations += 1
            self._best_guess = mid
            if self._iterations == self._max_iterations:
                raise BisectionError("no solution reached after max iterations")
        return mid

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def remaining_diff(self) -> float:
        return self._remaining_diff

    @property
    def best_guess(self) -> float:
        return self._best_guess

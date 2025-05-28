import numpy as np
import jax.numpy as jnp
from functools import partial
import jax


@jax.jit
def apply_poisson(U, h=None):
    """Apply the 3D poisson operator to U."""
    alpha = len(U.shape)
    x = jnp.empty_like(U)

    if h is None:
        h = 1 / U.shape[0]

    if alpha == 3:
        x = x.at[:, :, 0].set(U[:, :, 0])
        x = x.at[:, 0, :].set(U[:, 0, :])
        x = x.at[0, :, :].set(U[0, :, :])
        x = x.at[:, :, -1].set(U[:, :, -1])
        x = x.at[:, -1, :].set(U[:, -1, :])
        x = x.at[-1, :, :].set(U[-1, :, :])
        x = x.at[1:-1, 1:-1, 1:-1].set((-6.0 * U[1:-1, 1:-1, 1:-1] +
                               U[:-2, 1:-1, 1:-1] +
                               U[2:, 1:-1, 1:-1] +
                               U[1:-1, :-2, 1:-1] +
                               U[1:-1, 2:, 1:-1] +
                               U[1:-1, 1:-1, :-2] +
                               U[1:-1, 1:-1, 2:]) / (h * h))
    else:
        raise ValueError('residual: invalid dimension')

    return x


#### RESTRICTION FUNCTIONS

def restriction(A):
    """
        applies simple restriction to A
        @param A n x n matrix
        @return (n//2 +1, n//2 + 1) matrix
    """
    # indicator for Dimension
    alpha = len(A.shape)
    # initialize result with respect to the wanted shape
    ret = jnp.empty(np.array(A.shape) // 2 + 1)
    # Index of the second to the last element to mention in ret (depends on
    # the shape of A)
    end = ret.shape[0] - (A.shape[0] + 1) % 2

    # Case: Dimension 1
    if alpha == 3:
        ret = restriction_3D(A, ret, end)
    # Case: Error
    else:
        raise ValueError('restriction: invalid dimension')

    return ret

#@jit(nopython=True, fastmath=True)
@partial(jax.jit, static_argnames=['end'])
def restriction_3D(A, ret, end):
    # get every second element in A
    ret = ret.at[:end:, :end:, :end:].set(A[::2, ::2, ::2])
    # special case: inner borders
    ret = ret.at[:end, :end, -1].set(A[::2, ::2, -1])
    ret = ret.at[-1, :end, :end].set(A[-1, ::2, ::2])
    ret = ret.at[:end, -1, :end].set(A[::2, -1, ::2])
    # special case: outer borders
    ret=  ret.at[:end, -1, -1].set(A[::2, -1, -1])
    ret = ret.at[-1, :end, -1].set(A[-1, ::2, -1])
    ret = ret.at[-1, -1, :end].set(A[-1, -1, ::2])
    # special case: outer corner
    ret = ret.at[-1, -1, -1].set(A[-1, -1, -1])
    return ret


def weighted_restriction(A):
    # indicator for Dimension
    alpha = len(A.shape)
    # initialize result with respect to the wanted shape
    ret = restriction(A)

    # min length is 3
    assert(A.shape[0] >= 3)

    if alpha == 3:
        ret = weighted_restriction_3D(A, ret)
    else:
        raise ValueError('weighted restriction: invalid dimension')
    return ret

#@(nopython=True, fastmath=True)
@partial(jax.jit)
def weighted_restriction_3D(A, ret):
    # core
    ret = ret.at[1:-1, 1:-1, 1:-1].mul(8)
    # edges
    ret = ret.at[1:-1, 1:-1, 1:-1].add((
        A[2:-1:2, 2:-1:2, 1:-2:2] + A[2:-1:2, 2:-1:2, 3::2] +
        A[2:-1:2, 1:-2:2, 2:-1:2] + A[2:-1:2, 3::2, 2:-1:2] +
        A[1:-2:2, 2:-1:2, 2:-1:2] + A[3::2, 2:-1:2, 2:-1:2]) * 4)
    # more edges
    ret = ret.at[1:-1, 1:-1, 1:-1].add((
        A[2:-1:2, 1:-2:2, 3::2] + A[2:-1:2, 3::2, 1:-2:2] +
        A[2:-1:2, 1:-2:2, 1:-2:2] + A[2:-1:2, 3::2, 3::2] +
        A[1:-2:2, 2:-1:2, 3::2] + A[3::2, 2:-1:2, 1:-2:2] +
        A[1:-2:2, 2:-1:2, 1:-2:2] + A[3::2, 2:-1:2, 3::2] +
        A[1:-2:2, 3::2, 2:-1:2] + A[3::2, 1:-2:2, 2:-1:2] +
        A[1:-2:2, 1:-2:2, 2:-1:2] + A[3::2, 3::2, 2:-1:2]) * 2)
    # corners
    ret = ret.at[1:-1, 1:-1, 1:-1].add(
        A[3::2, 1:-2:2, 1:-2:2] + A[3::2, 3::2, 1:-2:2] +
        A[3::2, 3::2, 3::2] + A[3::2, 1:-2:2, 3::2] +
        A[1:-2:2, 1:-2:2, 1:-2:2] + A[1:-2:2, 1:-2:2, 3::2] +
        A[1:-2:2, 3::2, 3::2] + A[1:-2:2, 3::2, 1:-2:2])

    ret = ret = ret.at[1:-1, 1:-1, 1:-1].mul(1/64)
    return ret

### prolongation


@partial(jax.jit, static_argnames=['fine_shape'])
def prolongation(e, fine_shape):
    """
    This interpolates/ prolongates to a grid of fine_shape
    @param e
    @param fine_shape targeted shape
    @return grid with fine_shape
    """
    # indicator for Dimension
    alpha = len(e.shape)
    # initialize result with respect to the wanted shape
    w = jnp.zeros(fine_shape)
    # Index of the second to the last element to mention in e (depends on the
    # shape of w)
    end = e.shape[0] - (w.shape[0] + 1) % 2
    # Index of the second to the last element to mention in w (depends on the
    # shape of w)
    wend = w.shape[0] - (w.shape[0] + 1) % 2

    if alpha == 3:
        w = prolongation_3D(w, e, end, wend)

    # Case: Error
    else:
        raise ValueError("prolongation: invalid dimension")
    return w

@partial(jax.jit, static_argnames=['end','wend'])
def prolongation_3D(w, e, end, wend):
    # copy elements from e to w
    w = w.at[:-1:2, :-1:2, :-1:2].set(e[:-1, :-1, :-1])
    
    w = w.at[:-1:2, -1, -1].set(e[:-1, -1, -1])
    w= w.at[-1, :-1:2, -1].set(e[-1, :-1, -1])
    w= w.at[-1, -1, :-1:2].set(e[-1, -1, :-1])
    
    w= w.at[:-1:2, :-1:2, -1].set(e[:-1, :-1, -1])
    w= w.at[:-1:2, -1, :-1:2].set(e[:-1, -1, :-1])
    w= w.at[-1, :-1:2, :-1:2].set(e[-1, :-1, :-1])
    
    w= w.at[-1, -1, -1].set(e[-1, -1, -1])

    # interpolate elements horizontally
    w = w.at[:-1:2, 1:-1:2, :-1:2].set((
        e[:-1, : end - 1, :-1] + e[:-1, 1:end, :-1]
    ) / 2)
    w = w.at[:-1:2, -1, 1:-1:2].set((e[:-1, -1, : end - 1] + e[:-1, -1, 1:end]) / 2)
    w= w.at[:-1:2, :-1:2, 1:-1:2] .set((
        e[:-1, :-1, : end - 1] + e[:-1, :-1, 1:end]
    ) / 2)
    w = w.at[:-1:2, 1:-1:2, -1].set((e[:-1, : end - 1, -1] + e[:-1, 1:end, -1]) / 2)
    w = w.at[:-1:2, 1:-1:2, 1:-1:2].set((
        e[:-1, : end - 1, : end - 1] +
        e[:-1, 1:end, 1:end] +
        e[:-1, : end - 1, 1:end] +
        e[:-1, 1:end, : end - 1]) / 4)

    # special case
    w = w.at[-1, 1:-1:2, :-1:2].set((e[-1, : end - 1, :-1] + e[-1, 1:end, :-1]) / 2)
    w= w.at[-1, -1, 1:-1:2].set((e[-1, -1, : end - 1] + e[-1, -1, 1:end]) / 2)
    w= w.at[-1, :-1:2, 1:-1:2].set((e[-1, :-1, : end - 1] + e[-1, :-1, 1:end]) / 2)
    w= w.at[-1, 1:-1:2, -1].set((e[-1, : end - 1, -1] + e[-1, 1:end, -1]) / 2)
    w= w.at[-1, 1:-1:2, 1:-1:2].set((
        e[-1, : end - 1, : end - 1] +
        e[-1, 1:end, 1:end] +
        e[-1, : end - 1, 1:end] +
        e[-1, 1:end, : end - 1]
    ) / 4)

    # interpolate elements vertically
    w = w.at[1:-1:2, :-1:2, :-1:2].set((
        e[: end - 1, :-1, :-1] + e[1:end, :-1, :-1]
    ) / 2)
    w = w.at[1:-1:2, -1, :-1:2].set((e[: end - 1, -1, :-1] + e[1:end, -1, :-1]) / 2)
    w = w.at[1:-1:2, :-1:2, -1].set((e[: end - 1, :-1, -1] + e[1:end, :-1, -1]) / 2)
    w = w.at[1:-1:2, -1, -1].set((e[: end - 1, -1, -1] + e[1:end, -1, -1]) / 2)
    w = w.at[1:-1:2, -1, 1:-1:2].set((
        w[1:-1:2, -1, : wend - 1: 2] + w[1:-1:2, -1, 2:wend:2]
    ) / 2)
    w = w.at[1:-1:2, :-1:2, 1:-1:2].set((
        w[1:-1:2, :-1:2, : wend - 1: 2] + w[1:-1:2, :-1:2, 2:wend:2]
    ) / 2)
    w = w.at[1:-1:2, 1:-1:2, -1].set((
        w[1:-1:2, : wend - 1: 2, -1] + w[1:-1:2, 2:wend:2, -1]
    ) / 2)
    w = w.at[1:-1:2, 1:-1:2, :-1:2].set((
        w[1:-1:2, : wend - 1: 2, :-1:2] + w[1:-1:2, 2:wend:2, :-1:2]
    ) / 2)
    w = w.at[1:-1:2, 1:-1:2, 1:-1:2].set((
        w[1:-1:2, 1:-1:2, : wend - 1: 2] + w[1:-1:2, 1:-1:2, 2:wend:2]
    ) / 2)
    return w

#### multigrid cycles

from abc import abstractmethod

class AbstractCycle:
    def __init__(self, F, v1, v2, mu, l, eps=1e-8, h=None):
        self.v1 = v1
        self.v2 = v2
        self.mu = mu
        self.F = F
        self.l = l
        self.eps = eps
        if h is None:
            self.h = 1 / F.shape[0]
        else:
            self.h = h
        if (self.l == 0):
            self.l = int(np.log2(self.F.shape[0])) - 1
        # ceck if l is plausible
        if np.log2(self.F.shape[0]) < self.l:
            raise ValueError('false value of levels')

    def __call__(self, U):
        return self.do_cycle(self.F, U, self.l, self.h)

    @abstractmethod
    def _presmooth(self, F, U, h):
        pass

    @abstractmethod
    def _postsmooth(self, F, U, h):
        pass

    @abstractmethod
    def _compute_residual(self, F, U, h):
        pass

    @abstractmethod
    def _solve(self, F, U, h):
        pass

    @abstractmethod
    def norm(self, U):
        pass

    @abstractmethod
    def restriction(self, r):
        pass

    def _residual(self, U):
        return self._compute_residual(self.F, U, self.h)

    def _compute_correction(self, r, l, h):
        e = jnp.zeros_like(r)
        for _ in range(self.mu):
            e = self.do_cycle(r, e, l, h)
        return e

    def do_cycle(self, F, U, l, h):

        if l <= 1 or U.shape[0] <= 1:
            return self._solve(F, U, h)

        U = self._presmooth(F=F, U=U, h=h)

        r = self._compute_residual(F=F, U=U, h=h)

        r = self.restriction(r)
        
        e = self._compute_correction(r, l - 1, 2 * h)

        e = prolongation(e, U.shape)

        # correction
        U += e

        return self._postsmooth(F=F, U=U, h=h)


class PoissonCycle(AbstractCycle):
    def __init__(self, F, v1, v2, mu, l, eps=1e-8, h=None):
        super().__init__(F, v1, v2, mu, l, eps, h)

    def _presmooth(self, F, U, h=None):
        return GS_RB(
            F,
            U=U,
            h=h,
            max_iter=self.v1,
            eps=self.eps)

    def _postsmooth(self, F, U, h=None):
        return GS_RB(
            F,
            U=U,
            h=h,
            max_iter=self.v2,
            eps=self.eps)

    def _compute_residual(self, F, U, h):
        return F - apply_poisson(U, h)

    def _solve(self, F, U, h):
        return GS_RB(
            F=F,
            U=U,
            h=h,
            max_iter=100,
            eps=self.eps,
            norm_iter=5)

    def norm(self, U):
        residual = self._residual(U)
        return jnp.linalg.norm(residual)

    def restriction(self, r):
        return weighted_restriction(r)

    

def poisson_multigrid(F, U, l, v1, v2, mu, iter_cycle, eps=1e-6, h=None):
    """Implementation of MultiGrid iterations
       should solve AU = F
       A is poisson equation
       @param U n x n Matrix
       @param F n x n Matrix
       @param v1 Gauss Seidel iterations in pre smoothing
       @param v2 Gauss Seidel iterations in post smoothing
       @param mu iterations for recursive call
       @return x n vector
    """

    cycle = PoissonCycle(F, v1, v2, mu, l, eps, h)
    return multigrid(cycle, U, eps, iter_cycle)


def multigrid(cycle, U, eps, iter_cycle):

    # scale the epsilon with the number of gridpoints
    eps *= U.shape[0] * U.shape[0]
    for i in range(1, iter_cycle + 1):
        U = cycle(U)
        norm = cycle.norm(U)
      #  print(f"Residual has a L2-Norm of {norm:.4} after {i} MGcycle")
        if norm <= eps:
         #   print(
         #       f"converged after {i} cycles with {norm:.4} error")
            break
    return U


import jax
def GS_RB(
    F,
    U=None,
    h=None,
    max_iter=100,
    eps=1e-8,
    norm_iter=1000,
):
    """
    red-black
    Solve AU = F, the poisson equation.

    @param F n vector
    @param h is distance between grid points | default is 1/N
    @return U n vector
    """
    if U is None:
        U = jnp.zeros_like(F)
    if h is None:
        h = 1 / (U.shape[0])

    h2 = h * h

    if len(F.shape) == 3:
        sweep = sweep_3D
    else:
        raise ValueError("Wrong Shape!!!")

    norm = 0.0  # declarate norm so we can output later
    it = 0
    #Gauss-Seidel-Iterationen
    while it < max_iter:
        it += 1
        # check sometimes if solutions converges
        if it % norm_iter == 0:
            norm = jnp.linalg.norm(F - apply_poisson(U, h))
            if norm <= eps:
                break

        # red
        U= sweep(1, F, U, h2)
        # black
        U = sweep(0, F, U, h2)

 #   print(f"converged after {it} iterations with {norm:.4} error")

    return U


@partial(jax.jit, static_argnames=['color','h2'])
def sweep_3D(color, F, U, h2):
    """
    Do the sweeps.

    @param color 1 = red 0 for black
    @param h is distance between grid points
    """
    m, n, o = F.shape

    U = U.at[2:m - 1:2, 1:n - 1:2, 1 + color:o - 1:2].set((
        U[1:m - 2:2, 1:n - 1:2, 1 + color:o - 1:2] +
        U[3:m:2, 1:n - 1:2, 1 + color:o - 1:2] +
        U[2:m - 1:2, 0:n - 2:2, 1 + color:o - 1:2] +
        U[2:m - 1:2, 2:n:2, 1 + color:o - 1:2] +
        U[2:m - 1:2, 1:n - 1:2, color:o - 2:2] +
        U[2:m - 1:2, 1:n - 1:2, 2 + color:o:2] -
        F[2:m - 1:2, 1:n - 1:2, 1 + color:o - 1:2] * h2) / (6.0))

    U = U.at[1:m - 1:2, 1:n - 1:2, 2 - color:o - 1:2].set((
        U[0:m - 2:2, 1:n - 1:2, 2 - color:o - 1:2] +
        U[2:m:2, 1:n - 1:2, 2 - color:o - 1:2] +
        U[1:m - 1:2, 0:n - 2:2, 2 - color:o - 1:2] +
        U[1:m - 1:2, 2:n:2, 2 - color:o - 1:2] +
        U[1:m - 1:2, 1:n - 1:2, 1 - color:o - 2:2] +
        U[1:m - 1:2, 1:n - 1:2, 3 - color:o:2] -
        F[1:m - 1:2, 1:n - 1:2, 2 - color:o - 1:2] * h2) / (6.0))

    U = U.at[1:m - 1:2, 2:n - 1:2, 1 + color:o - 1:2].set((
        U[0:m - 2:2, 2:n - 1:2, 1 + color:o - 1:2] +
        U[2:m:2, 2:n - 1:2, 1 + color:o - 1:2] +
        U[1:m - 1:2, 1:n - 2:2, 1 + color:o - 1:2] +
        U[1:m - 1:2, 3:n:2, 1 + color:o - 1:2] +
        U[1:m - 1:2, 2:n - 1:2, color:o - 2:2] +
        U[1:m - 1:2, 2:n - 1:2, 2 + color:o:2] -
        F[1:m - 1:2, 2:n - 1:2, 1 + color:o - 1:2] * h2) / (6.0))

    U = U.at[2:m - 1:2, 2:n - 1:2, 2 - color:o - 1:2].set((
        U[1:m - 2:2, 2:n - 1:2, 2 - color:o - 1:2] +
        U[3:m:2, 2:n - 1:2, 2 - color:o - 1:2] +
        U[2:m - 1:2, 1:n - 2:2, 2 - color:o - 1:2] +
        U[2:m - 1:2, 3:n:2, 2 - color:o - 1:2] +
        U[2:m - 1:2, 2:n - 1:2, 1 - color:o - 2:2] +
        U[2:m - 1:2, 2:n - 1:2, 3 - color:o:2] -
        F[2:m - 1:2, 2:n - 1:2, 2 - color:o - 1:2] * h2) / (6.0))
    return U
    


def gauss_seidel(A, F, U=None, eps=1e-10, max_iter=100):
    """Implementation of Gauss Seidl iterations
       should solve AU = F
       @param A n x m Matrix
       @param F n vector
       @return n vector
    """
    raise
    n, *_ = A.shape
    if U is None:
        U = jnp.zeros_like(F)

    for _ in range(max_iter):
        U_next = jnp.zeros_like(U)
        for i in range(n):
            left = jnp.dot(A[i, :i], U_next[:i])
            right = jnp.dot(A[i, i + 1:], U[i + 1:])
            U_next[i] = (F[i] - left - right) / (A[i, i])

        U = U_next
        if np.linalg.norm(F - A @ U) < eps:
            break

    return U

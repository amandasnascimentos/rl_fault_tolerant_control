import numpy as np
from scipy.optimize import lsq_linear


def scale_to_bounds(u, u_min, u_max):
    """Reduz u pela mesma proporção até caber nos limites. Preserva a direção."""
    r = 1.0
    over = u > u_max
    under = u < u_min
    if np.any(over):
        r = max(r, np.max(u[over] / u_max[over]))
    if np.any(under):
        r = max(r, np.max(u[under] / u_min[under]))
    return u / r


class Allocator:
    """Interface comum.

    allocate(tau, eta=None) -> u
      tau: (6,) forças e torques desejados, em N e N·m
      eta: (8,) eficiência de cada propulsor (1 = saudável, 0 = falha)
      u:   (8,) comando de cada propulsor, em N
    """

    def __init__(self, B, u_min, u_max):
        self.B = np.asarray(B, float)
        self.u_min = np.asarray(u_min, float)
        self.u_max = np.asarray(u_max, float)

    def allocate(self, tau, eta=None):
        raise NotImplementedError


class PseudoInverse(Allocator):
    """Baseline: usa a matriz nominal e ignora falhas."""

    def __init__(self, B, u_min, u_max):
        super().__init__(B, u_min, u_max)
        self.Bp = np.linalg.pinv(self.B)

    def allocate(self, tau, eta=None):
        return scale_to_bounds(self.Bp @ np.asarray(tau, float), self.u_min, self.u_max)


class WeightedPseudoInverse(Allocator):
    """W-PINV com a matriz efetiva B @ diag(eta).
    Com eta = 1, coincide com a pseudo-inversa."""

    def __init__(self, B, u_min, u_max, eps=1e-9):
        super().__init__(B, u_min, u_max)
        self.eps = eps

    def allocate(self, tau, eta=None):
        n = self.B.shape[1]
        eta = np.ones(n) if eta is None else np.asarray(eta, float)
        Beff = self.B * eta[None, :]
        A = np.vstack([Beff, np.sqrt(self.eps) * np.eye(n)])
        b = np.concatenate([np.asarray(tau, float), np.zeros(n)])
        return lsq_linear(A, b, bounds=(self.u_min, self.u_max)).x


class BoundedLeastSquares(Allocator):
    """QP com limites: min ||B_eff u - tau||^2 + eps ||u||^2, sujeito a
    u_min <= u <= u_max. Propulsores com eta = 0 ficam forçados a zero."""

    def __init__(self, B, u_min, u_max, eps=1e-6):
        super().__init__(B, u_min, u_max)
        self.eps = eps

    def allocate(self, tau, eta=None):
        n = self.B.shape[1]
        eta = np.ones(n) if eta is None else np.asarray(eta, float)
        Beff = self.B * eta[None, :]
        A = np.vstack([Beff, np.sqrt(self.eps) * np.eye(n)])
        b = np.concatenate([np.asarray(tau, float), np.zeros(n)])
        return lsq_linear(A, b, bounds=(self.u_min, self.u_max)).x


METHODS = {
    "pinv": PseudoInverse,
    "wpinv": WeightedPseudoInverse,
    "qp": BoundedLeastSquares,
}


def make_allocator(name, B, u_min, u_max):
    if name not in METHODS:
        raise ValueError(f"Método desconhecido: {name}. Opções: {list(METHODS)}")
    return METHODS[name](B, u_min, u_max)
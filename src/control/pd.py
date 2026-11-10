import numpy as np


def euler_de_T(T):
    R = T[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PDController:
    """PD de posição e atitude com saturação por eixo.
    Entrada: pose 4x4 do mundo, referência de posição (mundo) e atitude (rad).
    Saída: tau = [Fx, Fy, Fz, Mx, My, Mz] no corpo, em N e N·m, limitado por tau_max.
    """

    def __init__(self, cfg):
        self.kp = np.asarray(cfg["kp"], float)
        self.kd = np.asarray(cfg["kd"], float)
        self.fz_ff = float(cfg["fz_ff"])
        self.dt = float(cfg["dt"])
        self.tau_max = np.asarray(cfg["tau_max"], float)
        self.reset()

    def reset(self):
        self._p_prev = None
        self._e_prev = None

    def compute(self, T, p_ref, rpy_ref=(0.0, 0.0, 0.0)):
        p = T[:3, 3]
        R = T[:3, :3]
        e = euler_de_T(T)

        if self._p_prev is None:
            v = np.zeros(3)
            w = np.zeros(3)
        else:
            v = (p - self._p_prev) / self.dt
            w = np.array([wrap(a - b) for a, b in zip(e, self._e_prev)]) / self.dt
        self._p_prev = p.copy()
        self._e_prev = e.copy()

        ep = np.asarray(p_ref, float) - p
        f_world = self.kp[:3] * ep - self.kd[:3] * v
        f_world[2] += self.fz_ff
        f_body = R.T @ f_world

        ea = np.array([wrap(a - b) for a, b in zip(rpy_ref, e)])
        m_body = self.kp[3:] * ea - self.kd[3:] * w

        tau = np.concatenate([f_body, m_body])
        return np.clip(tau, -self.tau_max, self.tau_max)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.holoocean_config import HoloOceanBlueROVWrapper  # noqa: E402
from control.allocator import GenericThrusterAllocator as ThrusterAllocator  # noqa: E402

def euler_de_T(T):
    R = T[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def testa_tau(env, tau, nome, allocator, passos=60):
    tau = np.array(tau, float)
    u = allocator.allocate(tau)

    env.reset()
    s = env.step(np.zeros(8))
    T0 = s["Pose"]

    for _ in range(passos):
        s = env.step(u)

    T1 = s["Pose"]
    dp = T0[:3, :3].T @ (T1[:3, 3] - T0[:3, 3])
    drpy = np.degrees([wrap(a - b) for a, b in zip(euler_de_T(T1), euler_de_T(T0))])

    print(f"\n{nome}")
    print(f"  tau pedido     = {np.round(tau, 2)}")
    print(f"  B @ u          = {np.round(allocator.B @ u, 2)}")
    print(f"  u enviado      = {np.round(u, 3)}")
    print(f"  dp_corpo (m)   = {np.round(dp, 4)}")
    print(f"  d_rpy (graus)  = {np.round(drpy, 2)}")

if __name__ == "__main__":
    allocator = ThrusterAllocator()
    casos = [
        ("surge +5 N",        [ 20, 0, 0, 0, 0, 0]),
        ("sway +5 N",         [ 0, 20, 0, 0, 0, 0]),
        ("heave +10 N",       [ 0, 0, 20, 0, 0, 0]),
        ("roll +1 N·m",       [ 0, 0, 0, 1, 0, 0]),
        ("pitch +1 N·m",      [ 0, 0, 0, 0, 1, 0]),
        ("yaw +1 N·m",        [ 0, 0, 0, 0, 0, 1]),
    ]
    
    with HoloOceanBlueROVWrapper() as env:
        for nome, tau in casos:
            testa_tau(env, tau, nome, allocator)
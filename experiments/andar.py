import sys
from pathlib import Path

import numpy as np
import holoocean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.allocator import GenericThrusterAllocator          # noqa: E402
from environment.holoocean_config import get_default_scenario   # noqa: E402

SCENARIO = get_default_scenario()
alloc = GenericThrusterAllocator()
B = alloc.B
PASSOS = 5 * 60


def euler_de_T(T):
    R = T[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


TESTES = [
    ("surge (frente)", [20,  0,  0,  0,  0,  0], "dp", 0, +1),
    ("sway (lado)",    [ 0, 20,  0,  0,  0,  0], "dp", 1, +1),
    ("heave (cima)",   [ 0,  0, 20,  0,  0,  0], "dp", 2, +1),
    ("roll",           [ 0,  0,  0,  1,  0,  0], "dr", 0, +1),
    ("pitch",          [ 0,  0,  0,  0,  1,  0], "dr", 1, +1),
    ("yaw",            [ 0,  0,  0,  0,  0,  1], "dr", 2, +1),
]


def roda_teste(env, nome, tau, tipo, idx, sinal_esperado):
    u = alloc.allocate(np.array(tau, float))

    env.reset()
    s = env.step(np.zeros(8))
    T0 = s["Pose"]
    p0, R0, e0 = T0[:3, 3].copy(), T0[:3, :3].copy(), euler_de_T(T0)

    e_prev = e0.copy()
    acc = np.zeros(3)          # rotação acumulada, passo a passo, sem wrap no total
    pico = np.zeros(2)
    for _ in range(PASSOS):
        s = env.step(u)
        e = euler_de_T(s["Pose"])
        acc += np.array([wrap(a - b) for a, b in zip(e, e_prev)])
        e_prev = e
        pico = np.maximum(pico, np.abs(np.degrees(e[:2])))

    T1 = s["Pose"]
    dp = R0.T @ (T1[:3, 3] - p0)
    de = np.degrees(acc)

    if tipo == "dp":
        medido, unidade, outros = dp[idx], "m", np.delete(dp, idx)
    else:
        medido, unidade, outros = de[idx], "graus", np.delete(de, idx)

    ok = np.sign(medido) == sinal_esperado

    print(f"\n--- {nome} ---")
    print(f"  max|u| = {np.max(np.abs(u)):.2f}")
    print(f"  eixo principal: {medido:+.3f} {unidade}  {'OK' if ok else 'SINAL INVERTIDO'}")
    print(f"  acoplamento (maior outro eixo): {np.max(np.abs(outros)):.3f} {unidade}")
    print(f"  dp_corpo = {np.round(dp, 3)} m")
    print(f"  d_rpy    = {np.round(de, 2)} graus")
    print(f"  pico roll/pitch = {np.round(pico, 2)} graus")
    return ok


if __name__ == "__main__":
    resultados = []
    with holoocean.make(scenario_cfg=SCENARIO) as env:
        for nome, tau, tipo, idx, sinal in TESTES:
            resultados.append(roda_teste(env, nome, tau, tipo, idx, sinal))

    print("\n=== Resumo ===")
    for (nome, *_), ok in zip(TESTES, resultados):
        print(f"  {nome:16s} {'OK' if ok else 'FALHOU'}")
    print(f"\nGraus de liberdade com sinal correto: {sum(resultados)}/6")
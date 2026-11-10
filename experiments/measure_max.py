"""Mede o limite de comando de cada propulsor, nos dois sentidos.

Ajusta o ganho linear em |cmd| <= 10 e marca o joelho onde a aceleração
deixa de acompanhar essa reta.

Uso (a partir de rl_fault_tolerant_control/):
    python experiments/measure_max.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.holoocean_config import HoloOceanBlueROVWrapper  # noqa: E402

DT = 1.0 / 60.0
PASSOS = 30                                        # 0,5 s por nível
COMANDOS = [5, 10, 15, 20, 22, 24, 26, 28, 30, 35, 40, 60, 100]
PROPULSORES = [0, 4]


def aceleracao_inicial(env, j, cmd):
    env.reset()
    s = env.step(np.zeros(8))
    p0 = s["Pose"][:3, 3].copy()
    u = np.zeros(8)
    u[j] = cmd
    ts, ds = [], []
    for i in range(PASSOS):
        s = env.step(u)
        ts.append((i + 1) * DT)
        ds.append(np.linalg.norm(s["Pose"][:3, 3] - p0))
    ts, ds = np.array(ts), np.array(ds)
    return 2 * np.sum(ds * ts**2) / np.sum(ts**4)


def mede_lado(env, j, sinal):
    cmds = [sinal * c for c in COMANDOS]
    acel = np.array([aceleracao_inicial(env, j, c) for c in cmds])
    cmds = np.array(cmds, float)

    # ganho linear: média de a/cmd em |cmd| <= 10
    lin = np.abs(cmds) <= 10
    ganho = np.mean(acel[lin] / cmds[lin])

    # joelho: primeiro nível em que a aceleração fica abaixo de 95% da reta
    pred = ganho * cmds
    desvio = np.abs(acel) < 0.95 * np.abs(pred)
    idx = np.where(desvio)[0]
    joelho = abs(cmds[idx[0]]) if idx.size else float("nan")

    platô = np.max(np.abs(acel))
    print(f"\nHO{j} lado {'+' if sinal > 0 else '-'}")
    print(f"{'comando':>8s} {'aceleração':>12s} {'a/cmd':>10s} {'reta':>10s}")
    for c, a, p in zip(cmds, acel, pred):
        print(f"{c:8.1f} {a:12.4f} {a / c:10.5f} {p:10.4f}")
    print(f"ganho linear = {ganho:.5f}   platô = {platô:.4f}   "
          f"joelho ≈ {joelho:.1f}   u_lim (platô/ganho) ≈ {platô / abs(ganho):.1f}")


with HoloOceanBlueROVWrapper() as env:
    for j in PROPULSORES:
        mede_lado(env, j, +1)
        mede_lado(env, j, -1)
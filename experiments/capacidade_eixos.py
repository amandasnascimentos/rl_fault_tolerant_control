"""Capacidade máxima de cada eixo, por programação linear.

Para cada eixo i, resolve:
    max  ±B[i, :] · u
    s.a. u_min <= u <= u_max  (valores do YAML)

Rodar a partir de rl_fault_tolerant_control/:
    python experiments/capacidade_eixos.py

Não usa o simulador: só a matriz B e os limites do YAML.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.allocator import GenericThrusterAllocator  # noqa: E402

geo = GenericThrusterAllocator()
B = geo.B
bounds = [(geo.u_min, geo.u_max)] * B.shape[1]
nomes = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
atual = geo.config["control"]["tau_max"]

print(f"limites de comando: u in [{geo.u_min}, {geo.u_max}]\n")
print(f"{'eixo':5s} {'máx +':>10s} {'máx -':>10s} {'YAML tau_max':>14s}")
print("-" * 45)

for i, nome in enumerate(nomes):
    c = B[i, :]
    # linprog minimiza, então para maximizar c·u usamos -c
    r_pos = linprog(-c, bounds=bounds, method="highs")
    r_neg = linprog(+c, bounds=bounds, method="highs")
    max_pos = -r_pos.fun
    max_neg = -r_neg.fun
    print(f"{nome:5s} {max_pos:10.2f} {max_neg:10.2f} {atual[i]:14.2f}")

print("\nObs.: cada eixo é avaliado isoladamente (os outros em zero).")
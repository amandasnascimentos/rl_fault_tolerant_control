import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.allocator import GenericThrusterAllocator  # noqa: E402
from allocation.allocators import make_allocator            # noqa: E402

geo = GenericThrusterAllocator()
B = geo.B
U_MIN = geo.u_min
U_MAX = geo.u_max

# (nome, tau, falhas)
CASOS = [
    ("surge 20 N, sem falha",          [20, 0, 0, 0, 0, 0],  []),
    ("surge 20 N, prop 4 falho",       [20, 0, 0, 0, 0, 0],  [4]),
    ("surge 20 N, props 4 e 5",        [20, 0, 0, 0, 0, 0],  [4, 5]),
    ("surge 150 N (saturação)",        [150, 0, 0, 0, 0, 0], []),
    ("surge 150 N, prop 4 falho",      [150, 0, 0, 0, 0, 0], [4]),
    ("surge 150 N, props 4 e 5",       [150, 0, 0, 0, 0, 0], [4, 5]),
    ("surge + yaw (saturação)",        [120, 0, 0, 0, 0, 3], []),
    ("heave 150 N (saturação)",        [0, 0, 150, 0, 0, 0], []),
    ("sway 80 N + surge 80 N",         [80, 80, 0, 0, 0, 0], []),
]

for metodo in ["pinv", "wpinv", "qp"]:
    alloc = make_allocator(metodo, B, [U_MIN] * 8, [U_MAX] * 8)
    print(f"\n== {metodo} ==")
    for nome, tau, falhas in CASOS:
        tau = np.array(tau, float)
        eta = np.ones(8)
        eta[falhas] = 0.0

        u = alloc.allocate(tau, eta)
        tau_real = B @ (eta * u)
        erro = np.linalg.norm(tau_real - tau)

        viola = np.any(u > U_MAX + 1e-6) or np.any(u < U_MIN - 1e-6)
        status = "VIOLA LIMITE" if viola else "ok"

        print(f"  {nome:30s} erro = {erro:9.4f}   max|u| = {np.max(np.abs(u)):7.2f}   {status}")
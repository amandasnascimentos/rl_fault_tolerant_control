import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.allocator import GenericThrusterAllocator  # noqa: E402
from allocation.allocators import make_allocator            # noqa: E402
from environment.holoocean_config import HoloOceanBlueROVWrapper  # noqa: E402

geo = GenericThrusterAllocator()
alloc = make_allocator("qp", geo.B, [geo.u_min] * 8, [geo.u_max] * 8)

# pedidos que forçam saturação em cada eixo
pedidos = {
    "surge 120 N":  [120, 0, 0, 0, 0, 0],
    "sway 120 N":   [0, 120, 0, 0, 0, 0],
    "heave 120 N":  [0, 0, 120, 0, 0, 0],
    "yaw 10 N·m":   [0, 0, 0, 0, 0, 10],
    "roll 10 N·m":  [0, 0, 0, 10, 0, 0],
    "pitch 10 N·m": [0, 0, 0, 0, 10, 0],
}

print(f"limites: u_min = {geo.u_min}, u_max = {geo.u_max}\n")
print("-- alocação estática (sem simulador) --")
for nome, tau in pedidos.items():
    u = alloc.allocate(np.array(tau, float))
    ok = np.all(u <= geo.u_max + 1e-6) and np.all(u >= geo.u_min - 1e-6)
    print(f"  {nome:14s} max|u| = {np.max(np.abs(u)):6.2f}   "
          f"{'ok' if ok else 'VIOLA LIMITE'}")

print("\n-- verificação no simulador (2 s por caso) --")
with HoloOceanBlueROVWrapper() as env:
    casos = [
        ("surge +120 N", [120, 0, 0, 0, 0, 0]),
        ("surge -120 N", [-120, 0, 0, 0, 0, 0]),
        ("heave +120 N", [0, 0, 120, 0, 0, 0]),
        ("heave -120 N", [0, 0, -120, 0, 0, 0]),
    ]
    for nome, tau in casos:
        env.reset()
        s = env.step(np.zeros(8))
        p0 = s["Pose"][:3, 3].copy()
        u = alloc.allocate(np.array(tau, float))
        for _ in range(120):
            s = env.step(u)
        d = s["Pose"][:3, 3] - p0
        print(f"  {nome}: u in [{u.min():7.2f}, {u.max():6.2f}]   "
              f"desloc x={d[0]:+.3f} y={d[1]:+.3f} z={d[2]:+.3f} m")
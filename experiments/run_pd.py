import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.allocator import GenericThrusterAllocator  # noqa: E402
from allocation.allocators import make_allocator            # noqa: E402
from control.pd import PDController, euler_de_T              # noqa: E402
from environment.holoocean_config import HoloOceanBlueROVWrapper  # noqa: E402

METODO = "qp"
DURACAO = 30.0                                  # s
REF_POS = np.array([12.0, 0.0, 0.0])           # deslocamento a partir da posição inicial (m)
REF_RPY = np.array([0.0, 0.0, 0.0])            # atitude de referência (rad)
PASSO_LOG = 0.1                                # s, intervalo de impressão
EIXOS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

geo = GenericThrusterAllocator()
cfg = geo.config["control"]
dt = cfg["dt"]
alloc = make_allocator(METODO, geo.B, [geo.u_min] * 8, [geo.u_max] * 8)
pd = PDController(cfg)

n = int(round(DURACAO / dt))
every = max(1, int(round(PASSO_LOG / dt)))

tau_max_obs = np.zeros(6)
u_pico = 0.0
passos_sat = 0
registro = []     # (t, pos, rpy, v_mundo, v_corpo, tau, u)

with HoloOceanBlueROVWrapper() as env:
    env.reset()
    s = env.step(np.zeros(8))
    p0 = s["Pose"][:3, 3].copy()
    ref = p0 + REF_POS
    pd.reset()

    p_prev = p0.copy()
    for k in range(n):
        T = s["Pose"]
        tau = pd.compute(T, ref, REF_RPY)
        u = alloc.allocate(tau)
        s = env.step(u)

        Tn = s["Pose"]
        p = Tn[:3, 3].copy()
        R = Tn[:3, :3]
        v_mundo = (p - p_prev) / dt      # diferença entre passos consecutivos
        v_corpo = R.T @ v_mundo
        p_prev = p.copy()

        t = (k + 1) * dt
        rpy = np.degrees(euler_de_T(Tn))
        m_u = float(np.max(np.abs(u)))

        tau_max_obs = np.maximum(tau_max_obs, np.abs(tau))
        u_pico = max(u_pico, m_u)
        if m_u >= 0.95 * geo.u_max:
            passos_sat += 1

        registro.append((t, p, rpy, v_mundo, v_corpo, tau, m_u))

print("t(s)   pos_x    pos_y    yaw(deg)  v_x(m/s)  v_y(m/s)  |v_corpo_xy|  tau_Fx  tau_Fy  tau_Mz  max|u|")
for i, (t, p, rpy, vm, vc, tau, m_u) in enumerate(registro):
    if i % every == 0:
        print(f"{t:4.2f}  {p[0]:+7.3f}  {p[1]:+7.3f}  {rpy[2]:+8.2f}  "
              f"{vm[0]:+8.3f}  {vm[1]:+8.3f}  {np.linalg.norm(vc[:2]):12.3f}  "
              f"{tau[0]:+6.2f}  {tau[1]:+6.2f}  {tau[5]:+6.3f}  {m_u:6.2f}")

print("\n=== Resumo ===")
print("  tau máximo por eixo: " +
      ", ".join(f"{nome}={v:.2f}" for nome, v in zip(EIXOS, tau_max_obs)))
print(f"  max|u| atingido: {u_pico:.2f}  (u_max = {geo.u_max})")
print(f"  passos com |u| >= 0.95 * u_max: {passos_sat} de {n}")
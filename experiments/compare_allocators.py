import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation.allocator import GenericThrusterAllocator  # noqa: E402
from allocation.allocators import make_allocator            # noqa: E402
from control.pd import PDController                         # noqa: E402
from environment.holoocean_config import HoloOceanBlueROVWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# CENÁRIO (altere aqui)
# ---------------------------------------------------------------------------
METODOS = ["pinv", "wpinv", "qp"]
REF_POS = np.array([8.0, 8.0, 0.0])            # deslocamento a partir da posição inicial (m)
REF_YAW_DEG = 45.0                             # yaw de referência (graus)
DURACAO = 30.0                                 # s
TOL = 0.1                                      # m, critério de convergência
LIMIAR_SAT = 0.95                              # fração de u_max que conta como saturado
FALHAS = [4]                                   # índices de propulsores falhos (vazio = sem falha)
EFICIENCIA_FALHA = 0.0                         # 0 = parado; 0.5 = metade da força

# ---------------------------------------------------------------------------
geo = GenericThrusterAllocator()
cfg = geo.config["control"]
n_passos = int(DURACAO / cfg["dt"])
REF_RPY = np.array([0.0, 0.0, np.radians(REF_YAW_DEG)])


def monta_eta():
    eta = np.ones(8)
    eta[FALHAS] = EFICIENCIA_FALHA
    return eta


def executa(env, metodo):
    alloc = make_allocator(metodo, geo.B, [geo.u_min] * 8, [geo.u_max] * 8)
    pd = PDController(cfg)
    eta = monta_eta()
    B_eff = geo.B * eta[None, :]

    env.reset()
    s = env.step(np.zeros(8))
    p0 = s["Pose"][:3, 3].copy()
    ref = p0 + REF_POS
    pd.reset()

    u_pico = 0.0
    passos_sat = 0
    t_conv = None
    erro_final = None
    erro_alloc = []        # ||B_eff u - tau|| por passo
    chatter = []           # ||u_k - u_{k-1}|| por passo
    u_prev = None

    for k in range(n_passos):
        T = s["Pose"]
        tau = pd.compute(T, ref, REF_RPY)
        u = alloc.allocate(tau, eta)
        s = env.step(eta * u)

        m_u = float(np.max(np.abs(u)))
        u_pico = max(u_pico, m_u)
        if m_u >= LIMIAR_SAT * geo.u_max:
            passos_sat += 1

        erro_alloc.append(float(np.linalg.norm(B_eff @ u - tau)))
        if u_prev is not None:
            chatter.append(float(np.linalg.norm(u - u_prev)))
        u_prev = u.copy()

        erro = float(np.linalg.norm(ref - s["Pose"][:3, 3]))
        if t_conv is None and erro < TOL:
            t_conv = (k + 1) * cfg["dt"]
        erro_final = erro

    erro_alloc = np.array(erro_alloc)
    chatter = np.array(chatter)

    return {
        "erro_final": erro_final,
        "t_conv": t_conv,
        "passos_sat": passos_sat,
        "tempo_sat": passos_sat * cfg["dt"],
        "u_pico": u_pico,
        "ealloc_med": float(np.mean(erro_alloc)),
        "ealloc_max": float(np.max(erro_alloc)),
        "chat_med": float(np.mean(chatter)),
        "chat_max": float(np.max(chatter)),
    }


def descreve_cenario():
    falha = f"propulsores falhos {FALHAS} (eficiência {EFICIENCIA_FALHA})" if FALHAS else "sem falha"
    return (f"ref = {REF_POS.tolist()} m, yaw = {REF_YAW_DEG}°, {falha}, "
            f"duração {DURACAO} s, tol. {TOL} m")


if __name__ == "__main__":
    resultados = {}
    with HoloOceanBlueROVWrapper() as env:
        for metodo in METODOS:
            print(f"Rodando {metodo}...")
            resultados[metodo] = executa(env, metodo)

    print("\n=== Comparação de alocadores ===")
    print(descreve_cenario() + "\n")

    print("-- Rastreamento e saturação --")
    cab = (f"{'método':8s} {'erro final (m)':>15s} {'t conv (s)':>11s} "
           f"{'passos sat.':>12s} {'tempo sat. (s)':>15s} {'max|u|':>8s}")
    print(cab)
    print("-" * len(cab))
    for metodo in METODOS:
        r = resultados[metodo]
        t_conv = f"{r['t_conv']:.2f}" if r["t_conv"] is not None else "não"
        print(f"{metodo:8s} {r['erro_final']:15.4f} {t_conv:>11s} "
              f"{r['passos_sat']:12d} {r['tempo_sat']:15.2f} {r['u_pico']:8.2f}")

    print("\n-- Erro de alocação ||B_eff u - tau|| (N e N·m) --")
    cab2 = f"{'método':8s} {'média':>10s} {'pico':>10s}"
    print(cab2)
    print("-" * len(cab2))
    for metodo in METODOS:
        r = resultados[metodo]
        print(f"{metodo:8s} {r['ealloc_med']:10.4f} {r['ealloc_max']:10.4f}")

    print("\n-- Chattering ||u_k - u_(k-1)|| (N) --")
    cab3 = f"{'método':8s} {'média':>10s} {'pico':>10s}"
    print(cab3)
    print("-" * len(cab3))
    for metodo in METODOS:
        r = resultados[metodo]
        print(f"{metodo:8s} {r['chat_med']:10.4f} {r['chat_max']:10.4f}")
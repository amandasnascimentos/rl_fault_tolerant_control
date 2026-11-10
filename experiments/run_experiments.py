"""Roda todos os cenários × alocadores e salva os resultados em results/<data>/.

Uso (a partir de rl_fault_tolerant_control/):
    python experiments/run_experiments.py
"""
import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.experiment import (  # noqa: E402
    CENARIOS, LIMIAR_SAT, METODOS, TOL, carrega_geometria, run_one,
)
from environment.holoocean_config import HoloOceanBlueROVWrapper  # noqa: E402


def salva_run(caminho, s):
    nomes = (["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
              "ref_x", "ref_y", "ref_z", "erro_pos"]
             + [f"tau_{i}" for i in range(6)]
             + [f"u_{i}" for i in range(8)]
             + [f"eta_{i}" for i in range(8)]
             + ["ealloc", "chatter"])
    dados = np.column_stack([
        s["t"], s["pos"], s["rpy_deg"], s["ref"], s["erro_pos"],
        s["tau"], s["u"], s["eta"], s["ealloc"], s["chatter"],
    ])
    np.savetxt(caminho, dados, delimiter=",", header=",".join(nomes),
               comments="", fmt="%.6f")


def escreve_csv(caminho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


def metadados(geo, cfg):
    try:
        hver = importlib.metadata.version("holoocean")
    except Exception:
        hver = "n/a"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "n/a"
    return {
        "data": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "holoocean": hver,
        "git_commit": commit,
        "tol_m": TOL,
        "limiar_sat": LIMIAR_SAT,
        "dt_s": cfg["dt"],
        "u_min": geo.u_min,
        "u_max": geo.u_max,
        "k_n": geo.k_n,
        "kp": cfg["kp"],
        "kd": cfg["kd"],
        "fz_ff": cfg["fz_ff"],
        "tau_max": cfg["tau_max"],
        "metodos": METODOS,
        "cenarios": CENARIOS,
    }


def main():
    pasta = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (pasta / "runs").mkdir(parents=True)

    geo, cfg = carrega_geometria()
    resumos = []

    with HoloOceanBlueROVWrapper() as env:
        for nome, cen in CENARIOS.items():
            for metodo in METODOS:
                print(f"[{nome}] {metodo} ...", flush=True)
                serie, resumo = run_one(env, geo, cfg, nome, metodo, cen)
                salva_run(pasta / "runs" / f"{nome}__{metodo}.csv", serie)
                resumos.append(resumo)

    escreve_csv(pasta / "summary.csv", resumos)
    shutil.copy(ROOT / "src" / "config" / "bluerov2_config.yaml",
                pasta / "bluerov2_config.yaml")
    (pasta / "metadata.json").write_text(
        json.dumps(metadados(geo, cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResultados em: {pasta}")


if __name__ == "__main__":
    main()
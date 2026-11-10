"""Gera figuras, relatório Markdown e tabela LaTeX a partir de uma pasta de resultados.

Uso (a partir de rl_fault_tolerant_control/):
    python experiments/plot_results.py                      # pasta mais recente
    python experiments/plot_results.py results/20260101_120000
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METODOS = ["pinv", "wpinv", "qp"]
CORES = {"pinv": "tab:red", "wpinv": "tab:blue", "qp": "tab:green"}
N_U = 8

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLUNAS_RELATORIO = [
    ("erro_final_m", "erro final (m)", "{:.4f}"),
    ("t_conv_s", "t conv (s)", "{:.2f}"),
    ("passos_sat", "passos sat.", "{:.0f}"),
    ("tempo_sat_s", "tempo sat. (s)", "{:.2f}"),
    ("u_pico", "max\\|u\\| (N)", "{:.2f}"),
    ("ealloc_media", "ealloc média (N)", "{:.3f}"),
    ("ealloc_media_nao_sat", "ealloc média s/ sat. (N)", "{:.3f}"),
    ("chatter_media", "chatter médio (N)", "{:.3f}"),
]


def ultima_pasta():
    pastas = sorted(p for p in (ROOT / "results").iterdir() if p.is_dir())
    if not pastas:
        raise SystemExit("Nenhum resultado em results/. Rode run_experiments.py primeiro.")
    return pastas[-1]


def carrega_resumo(pasta):
    with open(pasta / "summary.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for linha in linhas:
        for k, v in linha.items():
            if k not in ("cenario", "metodo"):
                linha[k] = float(v)
    return linhas


def carrega_run(pasta, cen, metodo):
    return np.genfromtxt(pasta / "runs" / f"{cen}__{metodo}.csv",
                         delimiter=",", names=True)


def fig_trajetoria(pasta, cen, meta, fig_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for m in METODOS:
        d = carrega_run(pasta, cen, m)
        a1.plot(d["x"], d["y"], color=CORES[m], lw=1.2, label=m)
        a2.semilogy(d["t"], np.maximum(d["erro_pos"], 1e-6), color=CORES[m], lw=1.2, label=m)
    a1.plot(d["ref_x"][-1], d["ref_y"][-1], "k*", ms=10, label="referência")
    a1.set_aspect("equal")
    a1.set_xlabel("x (m)")
    a1.set_ylabel("y (m)")
    a1.set_title("Trajetória no plano XY")
    a1.legend()

    a2.axhline(meta["tol_m"], color="k", ls="--", lw=0.8, label=f"tol. {meta['tol_m']} m")
    a2.set_xlabel("tempo (s)")
    a2.set_ylabel("erro de posição (m)")
    a2.set_title("Erro de posição")
    a2.legend()

    fig.suptitle(cen.replace("_", " "))
    fig.savefig(fig_dir / f"trajetoria_{cen}.png")
    plt.close(fig)


def fig_comandos(pasta, cen, falhas, meta, fig_dir):
    fig, eixos = plt.subplots(len(METODOS), 1, figsize=(9, 8), sharex=True)
    for ax, m in zip(eixos, METODOS):
        d = carrega_run(pasta, cen, m)
        u = np.column_stack([d[f"u_{i}"] for i in range(N_U)])
        for i in range(N_U):
            falho = i in falhas
            ax.plot(d["t"], u[:, i], "--" if falho else "-", lw=1,
                    label=f"HO{i}" + (" (falho)" if falho else ""))
        ax.axhline(meta["u_max"], color="k", lw=0.8, ls=":")
        ax.axhline(meta["u_min"], color="k", lw=0.8, ls=":")
        ax.set_title(m, fontsize=10)
        ax.set_ylabel("u (N)")
    eixos[0].legend(ncol=8, fontsize=7, loc="upper right")
    eixos[-1].set_xlabel("tempo (s)")
    fig.suptitle(f"Comando enviado u por propulsor: {cen.replace('_', ' ')}")
    fig.savefig(fig_dir / f"comandos_{cen}.png")
    plt.close(fig)


def fig_serie(pasta, cen, coluna, ylabel, fig_dir, nome):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in METODOS:
        d = carrega_run(pasta, cen, m)
        ax.plot(d["t"], d[coluna], color=CORES[m], lw=1, label=m)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(cen.replace("_", " "))
    ax.legend()
    fig.savefig(fig_dir / f"{nome}_{cen}.png")
    plt.close(fig)


def fig_resumo(resumos, cenarios, fig_dir):
    metricas = [
        ("erro_final_m", "Erro final (m)"),
        ("t_conv_s", "Tempo de convergência (s)"),
        ("tempo_sat_s", "Tempo em saturação (s)"),
        ("ealloc_media_nao_sat", "Erro de alocação s/ sat. (N)"),
        ("chatter_media", "Chattering médio (N)"),
    ]
    fig, eixos = plt.subplots(1, len(metricas), figsize=(18, 4))
    x = np.arange(len(cenarios))
    w = 0.25
    for ax, (chave, rotulo) in zip(eixos, metricas):
        for j, m in enumerate(METODOS):
            vals = [next(r[chave] for r in resumos
                         if r["cenario"] == c and r["metodo"] == m) for c in cenarios]
            ax.bar(x + (j - 1) * w, vals, w, color=CORES[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in cenarios],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(rotulo, fontsize=9)
    eixos[0].legend()
    fig.savefig(fig_dir / "resumo_metricas.png")
    plt.close(fig)


def fmt(v, spec):
    if isinstance(v, float) and np.isnan(v):
        return "n.c."
    return spec.format(v)


def escreve_relatorio(pasta, resumos, meta):
    L = [f"# Resultados: {pasta.name}", ""]
    L += ["## Parâmetros", ""]
    L.append(f"- Passo de controle: {meta['dt_s']:.6f} s")
    L.append(f"- Limites de comando: u ∈ [{meta['u_min']}, {meta['u_max']}] N")
    L.append(f"- Critério de convergência: |erro| < {meta['tol_m']} m (primeiro cruzamento)")
    L.append(f"- Saturação: max|u| ≥ {meta['limiar_sat']} · u_max")
    L.append(f"- Ganhos PD: kp = {meta['kp']}, kd = {meta['kd']}, fz_ff = {meta['fz_ff']}")
    L.append(f"- tau_max = {meta['tau_max']}")
    L.append(f"- git: {meta['git_commit']} · holoocean: {meta['holoocean']} · python {meta['python']}")
    L.append("- n.c. = não convergiu dentro da duração")
    L.append("")
    L += ["## Cenários", ""]
    L.append("| cenário | referência (m) | yaw (°) | propulsores falhos (HO) | eficiência | duração (s) |")
    L.append("|---|---|---|---|---|---|")
    for nome, c in meta["cenarios"].items():
        falhas = c["falhas"] if c["falhas"] else "—"
        L.append(f"| {nome} | {c['ref']} | {c['yaw_deg']} | {falhas} | {c['eficiencia']} | {c['duracao']} |")
    L.append("")
    for nome in meta["cenarios"]:
        L += [f"## Resultados: {nome}", ""]
        L.append("| método | " + " | ".join(h for _, h, _ in COLUNAS_RELATORIO) + " |")
        L.append("|---|" + "---|" * len(COLUNAS_RELATORIO))
        for r in [r for r in resumos if r["cenario"] == nome]:
            celulas = [fmt(r[k], spec) for k, _, spec in COLUNAS_RELATORIO]
            L.append(f"| {r['metodo']} | " + " | ".join(celulas) + " |")
        L.append("")
    (pasta / "relatorio.md").write_text("\n".join(L), encoding="utf-8")


def escreve_tex(pasta, resumos):
    cols = [
        ("erro_final_m", r"$e_f$ (m)", "{:.4f}"),
        ("t_conv_s", r"$t_c$ (s)", "{:.2f}"),
        ("tempo_sat_s", r"$t_{sat}$ (s)", "{:.2f}"),
        ("ealloc_media_nao_sat", r"$\bar{e}_{a}$ s/sat. (N)", "{:.3f}"),
        ("chatter_media", r"$\bar{c}$ (N)", "{:.3f}"),
    ]
    L = [
        "% Requer \\usepackage{booktabs}",
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{ll" + "c" * len(cols) + "}",
        "\\toprule",
        "Cenário & Método & " + " & ".join(h for _, h, _ in cols) + " \\\\",
        "\\midrule",
    ]
    for r in resumos:
        cen = r["cenario"].replace("_", "\\_")
        celulas = [fmt(r[k], spec) for k, _, spec in cols]
        L.append(f"{cen} & {r['metodo']} & " + " & ".join(celulas) + " \\\\")
    L += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Métricas de rastreamento, saturação, erro de alocação e chattering.}",
        "\\label{tab:resultados}",
        "\\end{table}",
    ]
    (pasta / "tabela_resumo.tex").write_text("\n".join(L), encoding="utf-8")


def main():
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else ultima_pasta()
    meta = json.loads((pasta / "metadata.json").read_text(encoding="utf-8"))
    resumos = carrega_resumo(pasta)

    fig_dir = pasta / "figuras"
    fig_dir.mkdir(exist_ok=True)

    for cen, c in meta["cenarios"].items():
        fig_trajetoria(pasta, cen, meta, fig_dir)
        fig_comandos(pasta, cen, c["falhas"], meta, fig_dir)
        fig_serie(pasta, cen, "ealloc", "||B_eff·u − τ|| (N)", fig_dir, "erro_alocacao")
        fig_serie(pasta, cen, "chatter", "||u_k − u_(k−1)|| (N)", fig_dir, "chattering")
    fig_resumo(resumos, list(meta["cenarios"]), fig_dir)

    escreve_relatorio(pasta, resumos, meta)
    escreve_tex(pasta, resumos)

    print(f"Figuras: {fig_dir}")
    print(f"Relatório: {pasta / 'relatorio.md'}")
    print(f"Tabela LaTeX: {pasta / 'tabela_resumo.tex'}")


if __name__ == "__main__":
    main()
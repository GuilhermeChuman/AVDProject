"""
views.py
--------
Camada de visualizacao construida sobre metrics.py (ver especificacao
tecnica, secao 4: V1-V5).

Este modulo depende de matplotlib e numpy, mas NAO depende de PyQt5. Cada
funcao recebe dados (series de angulo, resultados de metrics.py) e devolve
um objeto matplotlib.figure.Figure, pronto para:
    - ser salvo em arquivo (fig.savefig(...)), ou
    - ser embutido num widget PyQt5 via FigureCanvasQTAgg(fig), exatamente
      como viewer.py ja faz hoje para o esqueleto 3D (self.scatter_figure).

Relacao com a especificacao (secao 4):
    plot_angle_with_cycles    -> V1 (gráfico de linhas com picos/vales)
    plot_rom_per_cycle         -> V2 (amplitude por ciclo ao longo da sessao)
    plot_rom_distribution       -> V3 (box plot + pontos, D x E)
    plot_small_multiples          -> V4 (miniaturas com escala comum)
    render_summary_table            -> V5 (tabela-resumo em figura, alem do
                                            SessionSummary em texto/CSV)

Uso tipico (fora do PyQt5, para testar rapido):
    import pandas as pd
    import metrics, views

    df = pd.read_csv("arquivo.csv")
    ar = metrics.smooth(df["kneeRangle"])
    al = metrics.smooth(df["kneeLangle"])
    cycles_r = metrics.detect_cycles(ar)
    fig = views.plot_angle_with_cycles(ar, cycles_r, label="Joelho direito")
    fig.savefig("joelho_direito.png")
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import metrics
from metrics import CycleDetectionResult, SessionSummary


# Paleta consistente em todas as visualizacoes (daltonismo-friendly: azul/laranja)
COLOR_RIGHT = "#1f77b4"   # azul   -> perna direita
COLOR_LEFT = "#d95f02"    # laranja -> perna esquerda
COLOR_THRESHOLD = "#999999"


# ---------------------------------------------------------------------------
# V1 — Gráfico de linha com picos e vales anotados
# ---------------------------------------------------------------------------

def plot_angle_with_cycles(
    angle_series: Sequence[float],
    cycles: CycleDetectionResult,
    time_axis: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
    label: str = "Ângulo do joelho",
    color: str = COLOR_RIGHT,
    ax: Optional[plt.Axes] = None,
    peak_label: str = "picos (extensão)",
    valley_label: str = "vales (flexão máx.)",
    title: Optional[str] = None,
) -> Figure:
    """
    V1: série temporal do ângulo com picos (extensão) e vales (flexão
    máxima) marcados, e linha tracejada opcional do limiar de disparo.

    Fundamentação (ver especificação, secção 4): a posição em eixo comum
    é o canal perceptivo mais preciso para leitura de valores quantitativos
    (Cleveland & McGill, 1985), e o gráfico de linha é o formato dominante
    e preferido por profissionais de saúde para séries temporais de marcha.

    Parameters
    ----------
    angle_series : série de ângulos (graus), idealmente já suavizada.
    cycles : resultado de metrics.detect_cycles() para esta série.
    time_axis : eixo X customizado (ex.: minutos). Se None, usa o índice
        da amostra.
    threshold : se informado, desenha uma linha horizontal tracejada no
        limiar de disparo do passo.
    label : rótulo da série, usado na legenda e no título.
    color : cor da linha principal.
    ax : eixo matplotlib existente para desenhar (permite compor com
        outras séries na mesma figura). Se None, cria uma figura nova.
    peak_label, valley_label : significado dos picos/vales na legenda
        (varia conforme a articulação, ver joints.py).
    title : título do gráfico (padrão: "<label> — ciclos de flexão-extensão").

    Returns
    -------
    matplotlib.figure.Figure
    """
    signal = np.asarray(angle_series, dtype=float)
    x = np.asarray(time_axis) if time_axis is not None else np.arange(len(signal))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(x, signal, color=color, linewidth=1.2, label=label)

    if len(cycles.peaks):
        ax.scatter(x[cycles.peaks], signal[cycles.peaks], color=color,
                   marker="^", s=40, zorder=5, label=f"{label} — {peak_label}")
    if len(cycles.valleys):
        ax.scatter(x[cycles.valleys], signal[cycles.valleys], color=color,
                   marker="v", s=40, zorder=5, edgecolor="black", linewidth=0.5,
                   label=f"{label} — {valley_label}")

    if threshold is not None:
        ax.axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=1,
                    label=f"Limiar de disparo ({threshold:.0f}°)")

    ax.set_xlabel("Tempo" if time_axis is not None else "Amostra")
    ax.set_ylabel("Ângulo (graus)")
    ax.set_title(title or f"{label} — ciclos de flexão-extensão")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_angle_bilateral(
    angle_right: Sequence[float],
    angle_left: Sequence[float],
    cycles_right: CycleDetectionResult,
    cycles_left: CycleDetectionResult,
    time_axis: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
    joint_label: str = "joelhos",
    peak_label: str = "picos (extensão)",
    valley_label: str = "vales (flexão máx.)",
) -> Figure:
    """
    Variante de V1 com as duas pernas sobrepostas no mesmo eixo, útil
    para comparação visual direta de amplitude e defasagem entre D/E
    (como nas Figuras 4-6 do artigo original).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_angle_with_cycles(angle_right, cycles_right, time_axis, threshold,
                            label="Direito", color=COLOR_RIGHT, ax=ax,
                            peak_label=peak_label, valley_label=valley_label)
    plot_angle_with_cycles(angle_left, cycles_left, time_axis, None,
                            label="Esquerdo", color=COLOR_LEFT, ax=ax,
                            peak_label=peak_label, valley_label=valley_label)
    ax.set_title(f"Ângulo — {joint_label} (D/E) — ciclos")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V2 — Amplitude por ciclo ao longo da sessão
# ---------------------------------------------------------------------------

def plot_rom_per_cycle(
    rom_right: Sequence[float],
    rom_left: Optional[Sequence[float]] = None,
    title: str = "Amplitude por ciclo ao longo da sessão",
) -> Figure:
    """
    V2: amplitude (ROM) de cada ciclo, na ordem em que ocorreram, com
    linha de tendência linear (ver metrics.fatigue_trend). Permite
    identificar fadiga (inclinação negativa) ou aprendizado/aquecimento
    (inclinação positiva) ao longo da sessão.

    Fundamentação (ver especificação, secção 4): responde à limitação
    apontada na literatura de que médias entre ciclos ocultam
    variabilidade — aqui cada ciclo é um ponto, não uma média.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    def _plot_side(values, color, label):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return
        x = np.arange(1, len(values) + 1)
        ax.plot(x, values, "o-", color=color, markersize=5, label=label)
        trend = metrics.fatigue_trend(values)
        if not np.isnan(trend["slope"]):
            y_fit = trend["slope"] * (x - 1) + trend["intercept"]
            ax.plot(x, y_fit, "--", color=color, alpha=0.6, linewidth=1,
                     label=f"{label} — tendência ({trend['slope']:+.2f}°/ciclo)")

    _plot_side(rom_right, COLOR_RIGHT, "Direito")
    if rom_left is not None:
        _plot_side(rom_left, COLOR_LEFT, "Esquerdo")

    ax.set_xlabel("Número do ciclo (passo)")
    ax.set_ylabel("Amplitude (ROM, graus)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V3 — Distribuição da amplitude (box plot) por perna e/ou sessão
# ---------------------------------------------------------------------------

def plot_rom_distribution(
    rom_groups: dict,
    title: str = "Distribuição da amplitude (ROM) por grupo",
) -> Figure:
    """
    V3: box plot comparando a distribuição de ROM por ciclo entre grupos
    (ex.: perna D vs. E; ou mesma perna em dias diferentes), com os
    pontos individuais sobrepostos para preservar a granularidade por
    ciclo.

    Fundamentação: box plot é um dos métodos recomendados por Cleveland
    & McGill (1985) para leitura comparativa de distribuições; a
    sobreposição de pontos evita esconder a variabilidade por trás do
    resumo estatístico.

    Parameters
    ----------
    rom_groups : dict {nome_do_grupo: array_de_rom}, ex.:
        {"Direito - Dia 1": [...], "Esquerdo - Dia 1": [...], ...}
    """
    labels = list(rom_groups.keys())
    data = [np.asarray(v, dtype=float) for v in rom_groups.values()]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    try:
        # matplotlib >= 3.9
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        # matplotlib < 3.9
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cccccc")
        patch.set_alpha(0.5)

    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values,
                   color="black", alpha=0.5, s=15, zorder=3)

    ax.set_ylabel("Amplitude (ROM, graus)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V1b — Várias articulações sobrepostas (seleção multi-articular)
# ---------------------------------------------------------------------------

def plot_multi_joint_timeseries(
    series: dict,
    time_axis: Optional[Sequence[float]] = None,
    title: str = "Ângulos das articulações selecionadas",
) -> Figure:
    """
    Sobrepõe, no mesmo eixo, o ângulo de várias articulações (todas em
    graus, então a escala é comparável). Útil para ver coordenação entre
    articulações (ex.: joelho x quadril x cotovelo durante a marcha).

    Parameters
    ----------
    series : dict {rótulo: série_de_ângulos}, ex.: {"Joelho (D)": [...], ...}
    time_axis : eixo X comum (ex.: minutos). Se None, usa o índice da amostra.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        x = np.asarray(time_axis) if time_axis is not None else np.arange(len(values))
        ax.plot(x, values, linewidth=1.1, label=label)
    ax.set_xlabel("Tempo (min)" if time_axis is not None else "Amostra")
    ax.set_ylabel("Ângulo (graus)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# V4 — Small multiples (miniaturas com escala comum)
# ---------------------------------------------------------------------------

def plot_small_multiples(
    sessions: dict,
    threshold: Optional[float] = None,
    ncols: int = 3,
) -> Figure:
    """
    V4: grade de miniaturas (uma por sessão/voluntário/dia), todas com a
    MESMA escala de eixo Y, para permitir comparação direta por posição
    alinhada — o canal de maior acurácia perceptiva segundo Cleveland &
    McGill (1985). Técnica descrita classicamente por Tufte.

    Parameters
    ----------
    sessions : dict {titulo_da_sessao: (angle_series, cycles)}, onde
        cycles é um metrics.CycleDetectionResult (ou None para omitir
        marcadores de pico/vale).
    threshold : limiar de disparo a desenhar em cada miniatura (opcional).
    ncols : número de colunas da grade.
    """
    n = len(sessions)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    all_values = np.concatenate([np.asarray(s[0], dtype=float) for s in sessions.values()])
    y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (title, (angle_series, cycles)) in zip(axes_flat, sessions.items()):
        signal = np.asarray(angle_series, dtype=float)
        ax.plot(signal, color=COLOR_RIGHT, linewidth=1)
        if cycles is not None:
            if len(cycles.peaks):
                ax.scatter(cycles.peaks, signal[cycles.peaks], color=COLOR_RIGHT, marker="^", s=18)
            if len(cycles.valleys):
                ax.scatter(cycles.valleys, signal[cycles.valleys], color=COLOR_RIGHT, marker="v", s=18)
        if threshold is not None:
            ax.axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=0.8)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # oculta eixos vazios, se houver
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Comparação entre sessões (escala de eixo comum)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# V5 — Tabela-resumo
# ---------------------------------------------------------------------------

def render_summary_table(summary: SessionSummary, session_label: str = "Sessão") -> Figure:
    """
    V5: renderiza o SessionSummary (de metrics.summarize_session) como
    uma figura de tabela, para exportação/impressão junto dos gráficos.
    Valores exatos são melhor transmitidos por tabela do que por gráfico
    (Cleveland & McGill, 1985) — complementa, não substitui, V1-V4.
    """
    rows = [
        ["Nº de ciclos", f"{summary.n_cycles_right}", f"{summary.n_cycles_left}"],
        ["ROM mediana (°)", f"{summary.rom_median_right:.1f}", f"{summary.rom_median_left:.1f}"],
        ["ROM IQR (°)",
         f"{summary.rom_iqr_right[0]:.1f}–{summary.rom_iqr_right[1]:.1f}",
         f"{summary.rom_iqr_left[0]:.1f}–{summary.rom_iqr_left[1]:.1f}"],
        ["ROM da sessão (°)", f"{summary.session_rom_right:.1f}", f"{summary.session_rom_left:.1f}"],
        ["Tendência (°/ciclo)", f"{summary.fatigue_right['slope']:+.2f}", f"{summary.fatigue_left['slope']:+.2f}"],
        ["Passos efetivos", f"{summary.effective_steps_right}", f"{summary.effective_steps_left}"],
    ]
    col_labels = ["Métrica", "Direito", "Esquerdo"]

    fig, ax = plt.subplots(figsize=(6, 0.5 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    ax.set_title(
        f"{session_label} — resumo\nÍndice de assimetria: {summary.asymmetry_index:.1f}%",
        fontsize=10, pad=20,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Execucao com dados reais (Elderly_Data)
# ---------------------------------------------------------------------------
#
# Uso:
#     python views.py                       # usa ./Elderly_Data e salva em ./figuras_reais
#     python views.py --data-dir CAMINHO --out-dir SAIDA
#     python views.py --sessao "M5 20220623"  # prefixo do arquivo (voluntario + data)
#
# Parametros calibrados com as 20 sessoes reais (ver documento de calibracao):
#     smooth_window = 3   (janela em AMOSTRAS; o FPS varia entre dias, ver aviso abaixo)
#     prominence    = 15  (graus; acima do piso de ruido, abaixo da amplitude dos passos)
#
# AVISO: o FPS e ~3.6-4.1 Hz em 21/06 e ~7-11 Hz em 23/06 e 28/06. Janelas em
# amostras equivalem a tempos diferentes em cada dia; compare dias com cautela
# ate que as janelas sejam convertidas para segundos.

SMOOTH_WINDOW = 3
PROMINENCE = 15.0


def _find_session(data_dir: str, prefix: str) -> str:
    import glob
    import os
    matches = sorted(glob.glob(os.path.join(data_dir, f"{prefix}*.csv")))
    if not matches:
        raise FileNotFoundError(f"Nenhum CSV com prefixo '{prefix}' em {data_dir}")
    return matches[0]


def _analyze(filepath: str):
    """Carrega um CSV real e devolve (df, fps, ar, al, cycles_r, cycles_l, rom_r, rom_l)."""
    from data_loader import load_session

    df, fps = load_session(filepath)
    ar = metrics.smooth(df["kneeRangle"].values, window=SMOOTH_WINDOW)
    al = metrics.smooth(df["kneeLangle"].values, window=SMOOTH_WINDOW)
    cycles_r = metrics.detect_cycles(ar, prominence=PROMINENCE)
    cycles_l = metrics.detect_cycles(al, prominence=PROMINENCE)
    rom_r = metrics.rom_per_cycle(ar, cycles_r)
    rom_l = metrics.rom_per_cycle(al, cycles_l)
    return df, fps, ar, al, cycles_r, cycles_l, rom_r, rom_l


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Gera as visualizacoes V1-V5 a partir de dados reais.")
    parser.add_argument("--data-dir", default="Elderly_Data", help="pasta com os CSVs (padrao: Elderly_Data)")
    parser.add_argument("--out-dir", default="figuras_reais", help="pasta de saida das figuras")
    parser.add_argument("--sessao", default="M5 20220623",
                        help="prefixo do arquivo da sessao principal (padrao: M5 em 23/06, Figura 6 do artigo)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="limiar de disparo (graus) a desenhar; omitido por padrao, pois nao consta no CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- V1, V2, V3, V5: sessao principal --------------------------------
    path = _find_session(args.data_dir, args.sessao)
    label = os.path.basename(path).replace(".csv", "")
    df, fps, ar, al, cycles_r, cycles_l, rom_r, rom_l = _analyze(path)
    t = df["elapsed_minutes"].values
    print(f"Sessao: {label} | {len(df)} linhas | {t[-1]:.1f} min | FPS ~{fps:.1f}")

    plot_angle_bilateral(ar, al, cycles_r, cycles_l, time_axis=t, threshold=args.threshold).savefig(
        os.path.join(args.out_dir, "v1_angulo_bilateral.png"), dpi=120)
    plot_rom_per_cycle(rom_r, rom_l).savefig(
        os.path.join(args.out_dir, "v2_rom_por_ciclo.png"), dpi=120)
    plot_rom_distribution({"Direito": rom_r, "Esquerdo": rom_l}, title=f"Distribuição da amplitude — {label}").savefig(
        os.path.join(args.out_dir, "v3_distribuicao.png"), dpi=120)

    summary = metrics.summarize_session(
        df["kneeRangle"].values, df["kneeLangle"].values,
        prominence=PROMINENCE, smooth_window=SMOOTH_WINDOW,
        threshold=args.threshold if args.threshold is not None else 90.0,
    )
    render_summary_table(summary, session_label=label).savefig(
        os.path.join(args.out_dir, "v5_tabela.png"), dpi=120)

    # ---- V4: mesmo voluntario em dias diferentes -------------------------
    volunteer = os.path.basename(path).split()[0]
    import glob
    files = sorted(glob.glob(os.path.join(args.data_dir, f"{volunteer} *.csv")),
                   key=lambda f: os.path.basename(f).split()[1])
    sessions = {}
    for f in files:
        stamp = os.path.basename(f).split()[1]
        day = f"{stamp[6:8]}/{stamp[4:6]}"
        d, _, a, _, c, _, _, _ = _analyze(f)
        sessions[f"{volunteer} - {day}"] = (a, c)
    if len(sessions) >= 1:
        plot_small_multiples(sessions, threshold=args.threshold, ncols=min(3, len(sessions))).savefig(
            os.path.join(args.out_dir, "v4_small_multiples.png"), dpi=120)

    print(f"Figuras salvas em: {os.path.abspath(args.out_dir)}")

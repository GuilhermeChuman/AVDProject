"""
metrics.py
----------
Modulo de calculo de amplitude de movimento (ROM - Range of Motion) a partir
de series temporais de angulos articulares (ex.: kneeRangle / kneeLangle do
viewer.py).

Este modulo NAO depende de PyQt5, matplotlib ou qualquer componente de
interface grafica. Ele recebe arrays/Series simples (numpy ou pandas) e
devolve numeros, arrays ou DataFrames. Isso permite:
    - testar as funcoes isoladamente (testes unitarios com dados sinteticos);
    - reutilizar o mesmo codigo em notebooks, scripts de linha de comando
      ou em uma futura versao web, sem reescrever nada;
    - manter viewer.py (ou qualquer outra camada de interface) como o unico
      lugar que conhece PyQt5/matplotlib, importando este modulo apenas para
      obter os numeros a serem plotados.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    from scipy.signal import find_peaks, savgol_filter
    _HAS_SCIPY = True
except ImportError:  # scipy pode nao estar instalado em todo ambiente
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 1. Pre-processamento
# ---------------------------------------------------------------------------

def smooth(signal: Sequence[float], window: int = 5, method: str = "moving_average") -> np.ndarray:
    """
    Suaviza a serie de angulos antes da deteccao de picos/vales.

    O ruido de estimativa de pose (MediaPipe) pode gerar maximos e minimos
    locais espurios; suavizar reduz falsos ciclos na deteccao (ver secao 3
    e 7 da especificacao: "limitacoes declaradas").

    Parameters
    ----------
    signal : sequencia de angulos (graus), ja ordenada no tempo.
    window : tamanho da janela de suavizacao (numero de amostras).
    method : "moving_average" (padrao, sempre disponivel) ou
             "savgol" (Savitzky-Golay, requer scipy; preserva melhor
             picos/vales que a media movel).

    Returns
    -------
    np.ndarray com o mesmo comprimento do sinal de entrada.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return signal
    window = max(1, min(window, len(signal)))

    if method == "savgol":
        if not _HAS_SCIPY:
            raise ImportError("method='savgol' requer scipy instalado.")
        # polyorder deve ser menor que window; window deve ser impar
        win = window if window % 2 == 1 else window + 1
        win = min(win, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        win = max(win, 3)
        polyorder = min(2, win - 1)
        return savgol_filter(signal, window_length=win, polyorder=polyorder)

    # media movel centrada (padrao), com bordas replicadas
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# 2. Deteccao de ciclos (picos = extensao, vales = flexao maxima)
# ---------------------------------------------------------------------------

@dataclass
class CycleDetectionResult:
    """Resultado da deteccao de ciclos de um sinal de angulo."""
    peaks: np.ndarray    # indices dos picos (extensao / perna "em pe")
    valleys: np.ndarray  # indices dos vales (flexao maxima)


def detect_cycles(
    angle_series: Sequence[float],
    prominence: float = 10.0,
    min_distance: int = 1,
) -> CycleDetectionResult:
    """
    Detecta picos (extensao) e vales (flexao maxima) na serie de angulo
    do joelho, correspondendo aos ciclos de flexao-extensao da marcha
    estacionaria.

    Requer scipy (scipy.signal.find_peaks). A 'prominence' (proeminencia
    minima) evita que oscilacoes pequenas de ruido sejam contadas como
    ciclos — ajuste conforme o nivel de ruido observado nos dados.

    Parameters
    ----------
    angle_series : serie de angulos (graus), idealmente ja suavizada
        com `smooth()`.
    prominence : proeminencia minima (graus) exigida para considerar um
        ponto como pico/vale real. Valores tipicos: 5-20 graus.
    min_distance : distancia minima (em numero de amostras) entre dois
        picos/vales consecutivos, para evitar deteccoes duplicadas muito
        proximas.

    Returns
    -------
    CycleDetectionResult com os indices (posicoes no array) dos picos e
    vales encontrados.
    """
    if not _HAS_SCIPY:
        raise ImportError("detect_cycles requer scipy instalado (scipy.signal.find_peaks).")

    signal = np.asarray(angle_series, dtype=float)
    peaks, _ = find_peaks(signal, prominence=prominence, distance=min_distance)
    valleys, _ = find_peaks(-signal, prominence=prominence, distance=min_distance)
    return CycleDetectionResult(peaks=peaks, valleys=valleys)


# ---------------------------------------------------------------------------
# 3. Amplitude (ROM) por ciclo e por sessao
# ---------------------------------------------------------------------------

def rom_per_cycle(
    angle_series: Sequence[float],
    cycles: CycleDetectionResult,
) -> np.ndarray:
    """
    Calcula a amplitude de movimento (ROM) de cada ciclo, definida como a
    diferenca entre o pico mais proximo (extensao) e o vale (flexao
    maxima) que o segue ou precede.

    Implementacao: para cada vale, procura o pico mais proximo em tempo
    (antes ou depois) e calcula |angulo(pico) - angulo(vale)|. Essa
    abordagem segue a pratica consolidada na literatura de marcha de
    calcular excursoes articulares a partir de pontos-chave (picos e
    vales) identificados no ciclo (ver especificacao, secao 3 e 4 - V1/V2).

    Parameters
    ----------
    angle_series : serie de angulos original (ou suavizada) usada na
        deteccao de ciclos.
    cycles : resultado de `detect_cycles()`.

    Returns
    -------
    np.ndarray com uma amplitude (graus) por vale encontrado, na ordem
    temporal dos vales. Vazio se nao houver picos ou vales suficientes.
    """
    signal = np.asarray(angle_series, dtype=float)
    peaks, valleys = cycles.peaks, cycles.valleys
    if len(peaks) == 0 or len(valleys) == 0:
        return np.array([])

    roms = []
    for v in valleys:
        # pico mais proximo (em indice) a este vale
        nearest_peak = peaks[np.argmin(np.abs(peaks - v))]
        rom = abs(signal[nearest_peak] - signal[v])
        roms.append(rom)
    return np.array(roms)


def session_rom(angle_series: Sequence[float]) -> float:
    """
    Amplitude global da sessao: diferenca entre o angulo maximo e minimo
    observados. Metrica secundaria e sensivel a outliers — preferir
    `rom_per_cycle()` para analise detalhada (ver especificacao, secao 3).
    """
    signal = np.asarray(angle_series, dtype=float)
    if len(signal) == 0:
        return float("nan")
    return float(np.nanmax(signal) - np.nanmin(signal))


# ---------------------------------------------------------------------------
# 4. Assimetria entre pernas
# ---------------------------------------------------------------------------

def asymmetry_index(rom_right: Sequence[float], rom_left: Sequence[float]) -> float:
    """
    Indice de assimetria (IA) entre a amplitude da perna direita e
    esquerda, baseado nas medianas de ROM por ciclo de cada lado:

        IA = (mediana(ROM_D) - mediana(ROM_E)) /
             (0.5 * (mediana(ROM_D) + mediana(ROM_E))) * 100

    IA > 0 indica maior amplitude na perna direita; IA < 0, na esquerda.
    Usa mediana (nao media) para reduzir sensibilidade a ciclos espurios
    remanescentes apos a deteccao.

    Returns
    -------
    float (percentual). NaN se nao houver dados suficientes em algum lado.
    """
    r = np.asarray(rom_right, dtype=float)
    l = np.asarray(rom_left, dtype=float)
    if len(r) == 0 or len(l) == 0:
        return float("nan")
    med_r, med_l = np.nanmedian(r), np.nanmedian(l)
    denom = 0.5 * (med_r + med_l)
    if denom == 0:
        return float("nan")
    return float((med_r - med_l) / denom * 100)


# ---------------------------------------------------------------------------
# 5. Tendencia ao longo da sessao (ex.: fadiga)
# ---------------------------------------------------------------------------

def fatigue_trend(rom_values: Sequence[float]) -> dict:
    """
    Ajusta uma regressao linear simples de ROM_c (eixo Y) contra o
    indice do ciclo (eixo X), para detectar tendencia de queda (fadiga)
    ou de aumento (aprendizado/aquecimento) ao longo da sessao.

    Returns
    -------
    dict com:
        slope      : inclinacao (graus de ROM por ciclo). Negativo sugere
                     reducao de amplitude ao longo da sessao.
        intercept  : intercepto da reta.
        r_squared  : coeficiente de determinacao (ajuste da reta).
    Retorna valores NaN se houver menos de 2 pontos.
    """
    y = np.asarray(rom_values, dtype=float)
    n = len(y)
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan")}

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared)}


# ---------------------------------------------------------------------------
# 6. Passos efetivos (reconhecidos pelo limiar do sistema principal)
# ---------------------------------------------------------------------------

def effective_steps(angle_series: Sequence[float], cycles: CycleDetectionResult, threshold: float = 90.0) -> int:
    """
    Conta quantos vales (flexoes maximas) ultrapassaram o limiar de
    disparo configurado no sistema principal (ex.: 90 graus), ou seja,
    quantos ciclos teriam efetivamente avancado a navegacao no Street
    View, distinguindo "movimento gravado" de "passo reconhecido" (ver
    especificacao, secao 3 - "passo efetivo").

    Parameters
    ----------
    angle_series : serie de angulos original (ou suavizada).
    cycles : resultado de `detect_cycles()`.
    threshold : limiar de disparo em graus (quanto MENOR o angulo no
        vale, MAIS flexionado o joelho; um "passo" ocorre quando o
        angulo no vale fica ABAIXO do limiar).

    Returns
    -------
    int: numero de vales com angulo < threshold.
    """
    signal = np.asarray(angle_series, dtype=float)
    if len(cycles.valleys) == 0:
        return 0
    return int(np.sum(signal[cycles.valleys] < threshold))


# ---------------------------------------------------------------------------
# 7. Resumo de sessao (base da tabela V5 da especificacao)
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    n_cycles_right: int
    n_cycles_left: int
    rom_median_right: float
    rom_median_left: float
    rom_iqr_right: tuple
    rom_iqr_left: tuple
    session_rom_right: float
    session_rom_left: float
    asymmetry_index: float
    fatigue_right: dict
    fatigue_left: dict
    effective_steps_right: int
    effective_steps_left: int


def summarize_session(
    angle_right: Sequence[float],
    angle_left: Sequence[float],
    prominence: float = 10.0,
    min_distance: int = 1,
    threshold: float = 90.0,
    smooth_window: int = 5,
) -> SessionSummary:
    """
    Funcao de conveniencia que encadeia todas as etapas (suavizacao,
    deteccao de ciclos, ROM por ciclo, assimetria, fadiga e passos
    efetivos) e devolve um resumo pronto para popular a tabela V5 da
    especificacao tecnica.

    Esta e a funcao que a camada de interface (viewer.py ou outra)
    normalmente vai chamar; as funcoes individuais acima ficam
    disponiveis para uso isolado/testes.
    """
    ar = smooth(angle_right, window=smooth_window)
    al = smooth(angle_left, window=smooth_window)

    cycles_r = detect_cycles(ar, prominence=prominence, min_distance=min_distance)
    cycles_l = detect_cycles(al, prominence=prominence, min_distance=min_distance)

    rom_r = rom_per_cycle(ar, cycles_r)
    rom_l = rom_per_cycle(al, cycles_l)

    def iqr(values: np.ndarray) -> tuple:
        if len(values) == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(values, 25)), float(np.percentile(values, 75)))

    return SessionSummary(
        n_cycles_right=len(rom_r),
        n_cycles_left=len(rom_l),
        rom_median_right=float(np.nanmedian(rom_r)) if len(rom_r) else float("nan"),
        rom_median_left=float(np.nanmedian(rom_l)) if len(rom_l) else float("nan"),
        rom_iqr_right=iqr(rom_r),
        rom_iqr_left=iqr(rom_l),
        session_rom_right=session_rom(ar),
        session_rom_left=session_rom(al),
        asymmetry_index=asymmetry_index(rom_r, rom_l),
        fatigue_right=fatigue_trend(rom_r),
        fatigue_left=fatigue_trend(rom_l),
        effective_steps_right=effective_steps(ar, cycles_r, threshold=threshold),
        effective_steps_left=effective_steps(al, cycles_l, threshold=threshold),
    )


# ---------------------------------------------------------------------------
# Demonstracao / verificacao rapida (nao substitui testes unitarios formais)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sinal sintetico simulando 5 ciclos de flexao-extensao do joelho,
    # com ruido, para checar visualmente se as funcoes se comportam
    # como esperado.
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10 * np.pi, 500)
    synthetic_right = 150 + 30 * (0.5 * (1 - np.cos(t))) + rng.normal(0, 2, size=t.shape)
    synthetic_left = 150 + 25 * (0.5 * (1 - np.cos(t + 0.3))) + rng.normal(0, 2, size=t.shape)

    summary = summarize_session(synthetic_right, synthetic_left)
    print("Resumo da sessao (dados sinteticos):")
    print(f"  Ciclos D/E: {summary.n_cycles_right} / {summary.n_cycles_left}")
    print(f"  ROM mediana D/E: {summary.rom_median_right:.1f} / {summary.rom_median_left:.1f} graus")
    print(f"  IQR D: {summary.rom_iqr_right}  IQR E: {summary.rom_iqr_left}")
    print(f"  Indice de assimetria: {summary.asymmetry_index:.1f}%")
    print(f"  Tendencia (slope) D/E: {summary.fatigue_right['slope']:.2f} / {summary.fatigue_left['slope']:.2f}")
    print(f"  Passos efetivos D/E: {summary.effective_steps_right} / {summary.effective_steps_left}")

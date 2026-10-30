"""
joints.py
---------
Catalogo de articulacoes e calculo de angulos articulares a partir das
coordenadas 3D do rastreamento corporal (colunas como `r_kneeX`, `l_hipY`...).

Por que este modulo existe
--------------------------
O CSV gravado pelo aplicativo traz apenas dois angulos prontos
(`kneeLangle`, `kneeRangle`), mas guarda as coordenadas X/Y/Z de ombro,
cotovelo, punho, quadril, joelho e tornozelo dos dois lados. Aqui o angulo de
qualquer articulacao e calculado a partir de tres pontos (proximal, articulacao,
distal), com a mesma definicao do angulo interno do joelho:

    angulo = angulo entre os vetores (proximal - articulacao) e (distal - articulacao)

Validacao: para o joelho (quadril-joelho-tornozelo) o angulo recalculado em 3D
concorda com `kneeLangle`/`kneeRangle` (correlacao ~0,9996; erro medio ~0,5 grau,
que e o arredondamento para inteiro da coluna original).

Nao depende de Streamlit, PyQt5 ou matplotlib; apenas numpy, pandas e metrics.py.

Para incluir outra articulacao, basta acrescentar uma entrada em JOINTS.
Observacao: o tornozelo nao esta disponivel porque o CSV nao tem ponto do pe
(precisaria de um terceiro ponto distal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import metrics

# lado -> (rotulo, prefixo das colunas)
SIDES = {
    "right": ("Direito", "r_"),
    "left": ("Esquerdo", "l_"),
}


@dataclass(frozen=True)
class Joint:
    key: str
    label: str
    landmarks: tuple            # (proximal, articulacao, distal), sem prefixo de lado
    peak_label: str             # significado dos picos (angulo maximo)
    valley_label: str           # significado dos vales (angulo minimo)
    default_prominence: float   # proeminencia inicial (graus) - ponto de partida, nao calibrado
    steps_applicable: bool = False   # "passo efetivo" so faz sentido para o joelho
    native_columns: Optional[dict] = None   # {"right": col, "left": col} se o CSV ja traz o angulo


JOINTS = {
    "knee": Joint(
        key="knee", label="Joelho",
        landmarks=("hip", "knee", "ankle"),
        peak_label="picos (extensão)", valley_label="vales (flexão máx.)",
        default_prominence=15.0, steps_applicable=True,
        native_columns={"right": "kneeRangle", "left": "kneeLangle"},
    ),
    "elbow": Joint(
        key="elbow", label="Cotovelo",
        landmarks=("shoulder", "elbow", "wrist"),
        peak_label="picos (extensão)", valley_label="vales (flexão máx.)",
        default_prominence=15.0,
    ),
    "hip": Joint(
        key="hip", label="Quadril",
        landmarks=("shoulder", "hip", "knee"),
        peak_label="picos (quadril estendido)", valley_label="vales (flexão máx.)",
        default_prominence=15.0,
    ),
    "shoulder": Joint(
        key="shoulder", label="Ombro",
        landmarks=("hip", "shoulder", "elbow"),
        peak_label="picos (braço elevado)", valley_label="vales (braço junto ao corpo)",
        default_prominence=10.0,
    ),
}


# ---------------------------------------------------------------------------
# Angulos
# ---------------------------------------------------------------------------

def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angulo (graus) em `b` formado por a-b-c. Entradas: arrays (n, 3)."""
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", u, v) / norm
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _landmark_columns(landmark: str, prefix: str) -> list:
    return [f"{prefix}{landmark}{axis}" for axis in "XYZ"]


def required_columns(joint_key: str, side: str) -> list:
    """Colunas do CSV necessarias para calcular a articulacao/lado."""
    prefix = SIDES[side][1]
    cols = []
    for lm in JOINTS[joint_key].landmarks:
        cols += _landmark_columns(lm, prefix)
    return cols


def _has_native(df: pd.DataFrame, joint_key: str, side: str) -> bool:
    native = JOINTS[joint_key].native_columns
    return bool(native) and native[side] in df.columns


def is_available(df: pd.DataFrame, joint_key: str) -> bool:
    """True se os dois lados podem ser obtidos (coluna nativa ou coordenadas)."""
    for side in SIDES:
        if not (_has_native(df, joint_key, side) or set(required_columns(joint_key, side)) <= set(df.columns)):
            return False
    return True


def available_joints(df: pd.DataFrame) -> list:
    """Chaves das articulacoes que o CSV permite analisar (ordem de JOINTS)."""
    return [k for k in JOINTS if is_available(df, k)]


def joint_angle(df: pd.DataFrame, joint_key: str, side: str, prefer_native: bool = True) -> np.ndarray:
    """
    Serie temporal do angulo da articulacao no lado indicado ("right"/"left").

    Se `prefer_native` e o CSV trouxer o angulo pronto (joelho), usa a coluna
    original, preservando os resultados anteriores. Caso contrario, calcula a
    partir das coordenadas 3D. Falhas pontuais de rastreamento (NaN) sao
    interpoladas para nao quebrar a suavizacao/deteccao de ciclos.
    """
    if side not in SIDES:
        raise ValueError(f"lado invalido: {side!r}")
    joint = JOINTS[joint_key]

    if prefer_native and _has_native(df, joint_key, side):
        angle = df[joint.native_columns[side]].to_numpy(dtype=float)
    else:
        cols = [_landmark_columns(lm, SIDES[side][1]) for lm in joint.landmarks]
        missing = [c for group in cols for c in group if c not in df.columns]
        if missing:
            raise KeyError("Colunas ausentes: " + ", ".join(missing))
        a, b, c = (df[group].to_numpy(dtype=float) for group in cols)
        angle = compute_angle(a, b, c)

    if np.isnan(angle).any():
        angle = pd.Series(angle).interpolate(limit_direction="both").to_numpy()
    return angle


# ---------------------------------------------------------------------------
# Analise de uma articulacao (ambos os lados)
# ---------------------------------------------------------------------------

@dataclass
class JointAnalysis:
    joint: Joint
    raw: dict        # {"right": array, "left": array}  angulo bruto
    smoothed: dict   # angulo suavizado
    cycles: dict     # CycleDetectionResult por lado
    rom: dict        # ROM por ciclo por lado
    summary: metrics.SessionSummary


def analyze_joint(
    df: pd.DataFrame,
    joint_key: str,
    smooth_window: int = 3,
    prominence: Optional[float] = None,
    threshold: float = 90.0,
) -> JointAnalysis:
    """Encadeia suavizacao, deteccao de ciclos, ROM e resumo para uma articulacao."""
    joint = JOINTS[joint_key]
    prom = joint.default_prominence if prominence is None else float(prominence)

    raw = {s: joint_angle(df, joint_key, s) for s in SIDES}
    smoothed = {s: metrics.smooth(raw[s], window=int(smooth_window)) for s in SIDES}
    cycles = {s: metrics.detect_cycles(smoothed[s], prominence=prom) for s in SIDES}
    rom = {s: metrics.rom_per_cycle(smoothed[s], cycles[s]) for s in SIDES}
    summary = metrics.summarize_session(
        raw["right"], raw["left"],
        prominence=prom, smooth_window=int(smooth_window), threshold=float(threshold),
    )
    return JointAnalysis(joint, raw, smoothed, cycles, rom, summary)

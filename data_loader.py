"""
data_loader.py
--------------
Carrega uma sessao (CSV) de Elderly_Data SEM depender de PyQt5.

Reaproveita a mesma logica de tempo de load_csv() do viewer.py: o campo
`time` (HHMMSS) tem resolucao de 1 s e varios frames caem no mesmo segundo,
entao os frames sao distribuidos uniformemente dentro de cada segundo.
Isso e uma aproximacao (ver secao 7 da especificacao).
"""

import pandas as pd


def load_session(filepath):
    """
    Returns
    -------
    df  : DataFrame original + colunas `elapsed_seconds` e `elapsed_minutes`.
    fps : taxa de amostragem media estimada (frames por segundo).
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    counts = df["time"].value_counts()
    elapsed = []
    j = 0
    milliseconds = 0
    last_time = -1
    for csv_time in df["time"]:
        if csv_time == last_time:
            milliseconds += 1
        else:
            milliseconds = 0
            j += 1
        elapsed.append(j + milliseconds / counts[csv_time])
        last_time = csv_time

    df["elapsed_seconds"] = elapsed
    df["elapsed_minutes"] = df["elapsed_seconds"] / 60.0

    total_seconds = elapsed[-1] - elapsed[0]
    fps = len(df) / total_seconds if total_seconds > 0 else float("nan")
    return df, fps

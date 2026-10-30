"""
Dashboard Streamlit para análise de sessões de marcha / movimento.
Coloque este arquivo na mesma pasta de views.py, metrics.py, data_loader.py e joints.py.
Por padrão, procura os CSVs na pasta Elderly_Data.
Execute com: streamlit run dashboard.py

Novidade: é possível escolher a articulação analisada (joelho, cotovelo,
quadril ou ombro), uma por vez. Os ângulos das articulações que não vêm prontos no CSV são
calculados a partir das coordenadas 3D (ver joints.py).
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joints
import metrics
import views
from data_loader import load_session


st.set_page_config(
    page_title="Dashboard de Análise de Movimento",
    page_icon="",
    layout="wide",
)

# Estilo visual simples
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
        background: #f5f7fa; border: 1px solid #e4e8ef;
        padding: 14px 16px; border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dashboard de Análise de Movimento")
st.caption("Ângulos articulares, ciclos de flexão-extensão e amplitude (ROM) — escolha a articulação.")

DEFAULT_DATA_DIR = Path("Elderly_Data")


@st.cache_data(show_spinner=False)
def load_cached(path_str: str, mtime: float):
    """Carrega a sessão uma única vez por arquivo (mtime invalida o cache se o arquivo mudar)."""
    return load_session(path_str)


def fmt_joint(key: str) -> str:
    return joints.JOINTS[key].label


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configurações")
    data_dir_text = st.text_input("Pasta com os arquivos CSV", value=str(DEFAULT_DATA_DIR))
    data_dir = Path(data_dir_text).expanduser()
    threshold = st.number_input(
        "Limiar de referência do joelho (graus)",
        min_value=0.0,
        max_value=180.0,
        value=90.0,
        step=5.0,
        help="Linha de referência / passo efetivo. Só se aplica ao joelho. O valor padrão é 90°.",
    )
    smooth_window = st.number_input(
        "Janela de suavização",
        min_value=1,
        max_value=31,
        value=int(getattr(views, "SMOOTH_WINDOW", 3)),
        step=2,
        help="Número de amostras usado na suavização. Valores ímpares costumam ser convenientes.",
    )

# ---------------------------------------------------------------------------
# Seleção da sessão
# ---------------------------------------------------------------------------
st.subheader("Selecionar sessão")

csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
uploaded = st.file_uploader("Ou selecione um arquivo CSV do computador", type=["csv"])

selected_path = None
session_label = None
if csv_files:
    names_to_paths = {p.name: p for p in csv_files}
    selected_name = st.selectbox(
        "Arquivo da pasta de dados",
        options=list(names_to_paths.keys()),
        help="A seleção atualiza os indicadores e gráficos abaixo.",
    )
    selected_path = names_to_paths[selected_name]
elif uploaded is None:
    st.warning(
        f"Não encontrei arquivos CSV em `{data_dir}`. "
        "Informe a pasta correta na barra lateral ou envie um CSV abaixo."
    )

if uploaded is not None:
    # Arquivo enviado pelo usuário: gravar temporariamente para reutilizar data_loader.load_session.
    suffix = Path(uploaded.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        selected_path = Path(tmp.name)
    session_label = uploaded.name
elif selected_path is not None:
    session_label = selected_path.name

if selected_path is not None:
    try:
        df, fps = load_cached(str(selected_path), selected_path.stat().st_mtime)

        avail = joints.available_joints(df)
        if not avail:
            st.error(
                "O arquivo não contém colunas suficientes para nenhuma articulação "
                "(esperado: kneeLangle/kneeRangle ou coordenadas como r_kneeX, l_hipY...)."
            )
            st.stop()

        # -------------------------------------------------------------------
        # Seleção da articulação (uma por vez)
        # -------------------------------------------------------------------
        st.subheader("Selecionar articulação")
        k = st.radio(
            "Articulação a analisar",
            options=avail,
            index=avail.index("knee") if "knee" in avail else 0,
            format_func=fmt_joint,
            horizontal=True,
            help="Cada articulação é analisada nos dois lados (direito e esquerdo).",
        )
        j = joints.JOINTS[k]

        with st.sidebar:
            prominence = st.number_input(
                f"Proeminência — {j.label} (graus)",
                min_value=1.0,
                max_value=90.0,
                value=float(j.default_prominence),
                step=1.0,
                key=f"prom_{k}",
                help="Altura mínima de um pico/vale para contar como ciclo. Valor inicial, não calibrado.",
            )

        a = joints.analyze_joint(
            df, k,
            smooth_window=int(smooth_window),
            prominence=prominence,
            threshold=float(threshold),
        )
        s = a.summary

        if "elapsed_minutes" in df.columns:
            time_axis = df["elapsed_minutes"].to_numpy()
            duration = float(time_axis[-1]) if len(time_axis) else 0.0
        else:
            time_axis = None
            duration = (len(df) / float(fps) / 60) if fps else 0.0

        st.markdown(f"### Sessão: `{session_label}` — {j.label}")
        # c1, c2, c3, c4 = st.columns(4)
        # c1.metric("Amostras", f"{len(df):,}".replace(",", "."))
        # c2.metric("Duração aproximada", f"{duration:.2f} min")
        # c3.metric("Ciclos — direito", f"{s.n_cycles_right}")
        # c4.metric("Ciclos — esquerdo", f"{s.n_cycles_left}")

        st.divider()
        st.subheader(f"1. Ângulos — {j.label.lower()}")
        fig1 = views.plot_angle_bilateral(
            a.smoothed["right"], a.smoothed["left"],
            a.cycles["right"], a.cycles["left"],
            time_axis=time_axis,
            threshold=float(threshold) if j.steps_applicable else None,
            joint_label=j.label.lower(),
            peak_label=j.peak_label,
            valley_label=j.valley_label,
        )
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        left, right = st.columns(2)
        with left:
            st.subheader("2. Amplitude ROM por ciclo")
            fig2 = views.plot_rom_per_cycle(
                a.rom["right"], a.rom["left"],
                title=f"Amplitude por ciclo — {j.label.lower()}",
            )
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        with right:
            st.subheader("3. Distribuição da amplitude ROM")
            fig3 = views.plot_rom_distribution(
                {"Direito": a.rom["right"], "Esquerdo": a.rom["left"]},
                title=f"Distribuição da amplitude — {j.label} — {session_label}",
            )
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        st.divider()
        # st.subheader("4. Indicadores resumidos")
        # m1, m2, m3 = st.columns(3)
        # m1.metric("ROM mediana — direito", f"{s.rom_median_right:.1f}°")
        # m2.metric("ROM mediana — esquerdo", f"{s.rom_median_left:.1f}°")
        # m3.metric("Índice de assimetria", f"{s.asymmetry_index:.1f}%")

        rows = [
            ["Número de ciclos", s.n_cycles_right, s.n_cycles_left],
            ["ROM mediana (°)", s.rom_median_right, s.rom_median_left],
            ["ROM IQR inferior (°)", s.rom_iqr_right[0], s.rom_iqr_left[0]],
            ["ROM IQR superior (°)", s.rom_iqr_right[1], s.rom_iqr_left[1]],
            ["ROM da sessão (°)", s.session_rom_right, s.session_rom_left],
            ["Tendência (°/ciclo)", s.fatigue_right["slope"], s.fatigue_left["slope"]],
        ]
        if j.steps_applicable:
            rows.append(["Passos efetivos", s.effective_steps_right, s.effective_steps_left])
        table = pd.DataFrame(rows, columns=["Métrica", "Direito", "Esquerdo"])
        st.dataframe(table, hide_index=True, use_container_width=True)

        st.download_button(
            "Baixar tabela-resumo (CSV)",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"resumo_{Path(session_label).stem}_{k}.csv",
            mime="text/csv",
        )

        with st.expander("Visualizar dados carregados"):
            st.dataframe(df.head(1000), use_container_width=True)
            st.caption("Exibindo até as primeiras 1.000 linhas.")
    except Exception as exc:
        st.error(f"Não foi possível analisar o arquivo selecionado: {exc}")
        st.info(
            "Verifique se metrics.py, data_loader.py e joints.py estão na mesma pasta do dashboard "
            "e se o CSV segue o formato esperado pelo projeto."
        )

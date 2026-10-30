"""
Dashboard Streamlit para análise de sessões de marcha / movimento.
Coloque este arquivo na mesma pasta de views.py, metrics.py, data_loader.py e joints.py.
Por padrão, procura os CSVs na pasta Elderly_Data.
Execute com: streamlit run dashboard.py

Novidade: é possível selecionar UMA OU MAIS articulações (joelho, cotovelo,
quadril, ombro). Os ângulos das articulações que não vêm prontos no CSV são
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
    page_icon="📊",
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
st.caption("Ângulos articulares, ciclos de flexão-extensão e amplitude (ROM) — selecione uma ou mais articulações.")

DEFAULT_DATA_DIR = Path("Elderly_Data")
SIDE_LABELS = {s: v[0] for s, v in joints.SIDES.items()}          # right -> "Direito"
SIDE_SHORT = {"right": "D", "left": "E"}


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
        # Seleção de articulações
        # -------------------------------------------------------------------
        st.subheader("Selecionar articulações")
        selected = st.multiselect(
            "Articulações a analisar",
            options=avail,
            default=["knee"] if "knee" in avail else avail[:1],
            format_func=fmt_joint,
            help="Escolha uma ou mais. Cada articulação é analisada nos dois lados (direito e esquerdo).",
        )
        if not selected:
            st.info("Selecione ao menos uma articulação para ver os gráficos.")
            st.stop()

        with st.sidebar:
            with st.expander("Proeminência por articulação (graus)", expanded=False):
                st.caption("Altura mínima de um pico/vale para contar como ciclo. Valores iniciais, não calibrados.")
                prominences = {
                    k: st.number_input(
                        fmt_joint(k),
                        min_value=1.0,
                        max_value=90.0,
                        value=float(joints.JOINTS[k].default_prominence),
                        step=1.0,
                        key=f"prom_{k}",
                    )
                    for k in selected
                }

        # -------------------------------------------------------------------
        # Análise de cada articulação selecionada
        # -------------------------------------------------------------------
        analyses = {
            k: joints.analyze_joint(
                df, k,
                smooth_window=int(smooth_window),
                prominence=prominences[k],
                threshold=float(threshold),
            )
            for k in selected
        }

        if "elapsed_minutes" in df.columns:
            time_axis = df["elapsed_minutes"].to_numpy()
            duration = float(time_axis[-1]) if len(time_axis) else 0.0
        else:
            time_axis = None
            duration = (len(df) / float(fps) / 60) if fps else 0.0

        st.markdown(f"### Sessão: `{session_label}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Amostras", f"{len(df):,}".replace(",", "."))
        c2.metric("Duração aproximada", f"{duration:.2f} min")
        c3.metric("Taxa de amostragem", f"{fps:.1f} Hz" if fps == fps else "—")
        c4.metric("Articulações selecionadas", f"{len(selected)}")

        # -------------------------------------------------------------------
        # Comparação entre articulações (só faz sentido com 2 ou mais)
        # -------------------------------------------------------------------
        if len(selected) > 1:
            st.divider()
            st.subheader("Comparação entre articulações")

            side_choice = st.radio(
                "Lado exibido na sobreposição de ângulos",
                options=list(joints.SIDES.keys()),
                format_func=lambda s: SIDE_LABELS[s],
                horizontal=True,
            )
            series = {fmt_joint(k): a.smoothed[side_choice] for k, a in analyses.items()}
            fig_ts = views.plot_multi_joint_timeseries(
                series,
                time_axis=time_axis,
                title=f"Ângulos articulares — lado {SIDE_LABELS[side_choice].lower()}",
            )
            st.pyplot(fig_ts, use_container_width=True)
            plt.close(fig_ts)

            groups = {
                f"{fmt_joint(k)} ({SIDE_SHORT[s]})": a.rom[s]
                for k, a in analyses.items()
                for s in joints.SIDES
            }
            fig_cmp = views.plot_rom_distribution(
                groups, title=f"Distribuição da amplitude por articulação — {session_label}"
            )
            st.pyplot(fig_cmp, use_container_width=True)
            plt.close(fig_cmp)

        # -------------------------------------------------------------------
        # Detalhe por articulação (uma aba para cada)
        # -------------------------------------------------------------------
        st.divider()
        st.subheader("Detalhe por articulação")
        tabs = st.tabs([fmt_joint(k) for k in selected])

        for tab, k in zip(tabs, selected):
            a = analyses[k]
            j = a.joint
            s = a.summary
            with tab:
                use_threshold = float(threshold) if j.steps_applicable else None

                st.markdown(f"**1. Ângulos — {j.label.lower()}**")
                fig1 = views.plot_angle_bilateral(
                    a.smoothed["right"], a.smoothed["left"],
                    a.cycles["right"], a.cycles["left"],
                    time_axis=time_axis,
                    threshold=use_threshold,
                    joint_label=j.label.lower(),
                    peak_label=j.peak_label,
                    valley_label=j.valley_label,
                )
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

                left, right = st.columns(2)
                with left:
                    st.markdown("**2. Amplitude ROM por ciclo**")
                    fig2 = views.plot_rom_per_cycle(
                        a.rom["right"], a.rom["left"],
                        title=f"Amplitude por ciclo — {j.label.lower()}",
                    )
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                with right:
                    st.markdown("**3. Distribuição da amplitude ROM**")
                    fig3 = views.plot_rom_distribution(
                        {"Direito": a.rom["right"], "Esquerdo": a.rom["left"]},
                        title=f"Distribuição da amplitude — {j.label} — {session_label}",
                    )
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

                m1, m2, m3 = st.columns(3)
                m1.metric("ROM mediana — direito", f"{s.rom_median_right:.1f}°")
                m2.metric("ROM mediana — esquerdo", f"{s.rom_median_left:.1f}°")
                m3.metric("Índice de assimetria", f"{s.asymmetry_index:.1f}%")

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

        # -------------------------------------------------------------------
        # Tabela-resumo consolidada (todas as articulações selecionadas)
        # -------------------------------------------------------------------
        st.divider()
        st.subheader("Resumo consolidado")

        tidy_rows = []
        for k, a in analyses.items():
            s = a.summary
            for side in joints.SIDES:
                is_r = side == "right"
                tidy_rows.append({
                    "Articulação": a.joint.label,
                    "Lado": SIDE_LABELS[side],
                    "Ciclos": s.n_cycles_right if is_r else s.n_cycles_left,
                    "ROM mediana (°)": s.rom_median_right if is_r else s.rom_median_left,
                    "ROM IQR inferior (°)": (s.rom_iqr_right if is_r else s.rom_iqr_left)[0],
                    "ROM IQR superior (°)": (s.rom_iqr_right if is_r else s.rom_iqr_left)[1],
                    "ROM da sessão (°)": s.session_rom_right if is_r else s.session_rom_left,
                    "Tendência (°/ciclo)": (s.fatigue_right if is_r else s.fatigue_left)["slope"],
                    "Passos efetivos": (
                        (s.effective_steps_right if is_r else s.effective_steps_left)
                        if a.joint.steps_applicable else None
                    ),
                    "Assimetria D×E (%)": s.asymmetry_index,
                })
        summary_table = pd.DataFrame(tidy_rows)
        st.dataframe(
            summary_table.round(2),
            hide_index=True,
            use_container_width=True,
        )

        csv_bytes = summary_table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Baixar tabela-resumo (CSV)",
            data=csv_bytes,
            file_name=f"resumo_{Path(session_label).stem}.csv",
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

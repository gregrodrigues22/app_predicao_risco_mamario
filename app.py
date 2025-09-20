# ==============================================================
# Set up
# ==============================================================
import os
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import onnxruntime as ort

# --------------------------------------------------------------
# Configuração da página
# --------------------------------------------------------------
st.set_page_config(
    page_title="📈 Predição de Lesão de Mama",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
  /* Aumenta a largura da sidebar */
  [data-testid="stSidebar"] {
      width: 360px !important;      /* experimente 360–420px */
      min-width: 360px !important;
  }
  /* dá um respiro no conteúdo principal */
  .block-container {
      padding-left: 1.2rem;
      padding-right: 1.2rem;
  }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Utilitário: localizar primeiro arquivo existente (logo/foto)
# --------------------------------------------------------------
ASSETS = Path("assets")

def first_existing(*names, base=ASSETS):
    for n in names:
        p = base / n if base else Path(n)
        if p.exists():
            return p
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# --------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428); 
                padding: 36px; border-radius: 14px; margin-bottom:28px'>
        <h1 style='color: white; margin: 0;'>📊 Predição de Lesão Mamária</h1>
        <p style='color: #e8eef7; margin: 8px 0 0 0; font-size: 1.05rem;'>
            Explore a predição para tomada de decisão no point of care.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Esconde a lista padrão de páginas no topo da sidebar
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# Sidebar 
# --------------------------------------------------------------
with st.sidebar:
    if LOGO:
        st.image(str(LOGO), use_container_width=True)
    else:
        st.warning("Logo não encontrada em assets/.")
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    with st.expander("Predição no PoC", expanded=True):
        st.page_link("app.py", label="Predição de Lesão de Mama", icon="📈")

    with st.expander("Explicação do Modelo", expanded=True):
        st.page_link("pages/explain.py", label="Explicação do Modelo", icon="📙")

    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)

    st.subheader("Conecte-se")
    st.markdown(
        """
        - 💼 [LinkedIn](https://www.linkedin.com/in/gregorio-healthdata/)
        - ▶️ [YouTube](https://www.youtube.com/@Patients2Python)
        - 📸 [Instagram](https://www.instagram.com/patients2python/)
        - 🌐 [Site](https://patients2python.com.br/)
        - 🐙 [GitHub](https://github.com/gregrodrigues22)
        - 👥💬 [Comunidade](https://chat.whatsapp.com/CBn0GBRQie5B8aKppPigdd)
        - 🤝💬 [WhatsApp](https://patients2python.sprinthub.site/r/whatsapp-olz)
        - 🎓 [Escola](https://app.patients2python.com.br/browse)
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------------------
# CONTEÚDO PRINCIPAL DO APP
# --------------------------------------------------------------
st.title("Predição no PoC 📈🎯")
st.write("Preencha os campos abaixo com os valores correspondentes às variáveis utilizadas no modelo preditivo.")

# ------------------------- Modelo -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_final_proba4.onnx"

@st.cache_resource(show_spinner=True)
def load_session(model_path: Path):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.get_outputs()
    output_names = [o.name for o in outputs]   # ex.: ["label", "probabilities"]
    input_shape = sess.get_inputs()[0].shape   # ex.: [None, 20]
    return sess, input_name, output_names, input_shape

try:
    sess, INPUT_NAME, OUTPUT_NAMES, INPUT_SHAPE = load_session(MODEL_PATH)
except Exception as e:
    st.error(f"Falha ao carregar o modelo: {e}")
    st.stop()

# ------------------------- Schema -----------------------------
# ATENÇÃO: a ordem precisa ser exatamente a do treinamento
ORDERED_FEATURES = [
    "gender", "years_age",
    "fator_ca_ovario_final", "fator_hf_mama_final", "fator_genital_final",
    "fator_obesidade_final", "fator_sobrepeso_final", "fator_etilismo_final",
    "fator_tabagismo_final", "fator_alimentacao_final", "fator_sedentarismo_final",
    "fator_radiacao_final", "fator_anticoncepcao_final", "fator_dislipidemia_final",
    "fator_diabetes_final", "fator_hipertensao_final", "fator_gestacao_final",
    "fator_mental_final", "fator_neoplasias_final", "fator_menarca_final",
]

def normalize_age(age_years: int) -> float:
    # normaliza em 0–1 (ajuste se no treino foi outra escala)
    return max(0.0, min(1.0, age_years / 97.0))

# ------------------------- Widgets ----------------------------
st.subheader("Dados do Paciente")
c1, c2 = st.columns(2)
with c1:
    age_years = st.slider("Idade (anos)", 0, 100, 50)
    gender = st.radio("Sexo", options=[0, 1],
                      format_func=lambda v: "Feminino (0)" if v == 0 else "Masculino (1)",
                      horizontal=True)

def risk_radio(label: str, value: int = 0):
    return st.radio(label, options=[0, 1], index=value, horizontal=True)

cols = st.columns(2)
with cols[0]:
    fator_ca_ovario_final      = risk_radio("História de CA de ovário na família/pessoal?")
    fator_hf_mama_final        = risk_radio("História familiar de CA de mama?")
    fator_genital_final        = risk_radio("Fator genital?")
    fator_obesidade_final      = risk_radio("Obesidade?")
    fator_sobrepeso_final      = risk_radio("Sobrepeso?")
    fator_etilismo_final       = risk_radio("Etilismo?")
    fator_tabagismo_final      = risk_radio("Tabagismo?")
    fator_alimentacao_final    = risk_radio("Alimentação inadequada?")
    fator_sedentarismo_final   = risk_radio("Sedentarismo?")

with cols[1]:
    fator_radiacao_final       = risk_radio("Exposição à radiação?")
    fator_anticoncepcao_final  = risk_radio("Uso de anticoncepção?")
    fator_dislipidemia_final   = risk_radio("Dislipidemia?")
    fator_diabetes_final       = risk_radio("Diabetes?")
    fator_hipertensao_final    = risk_radio("Hipertensão?")
    fator_gestacao_final       = risk_radio("Gestação?")
    fator_mental_final         = risk_radio("Transtorno mental?")
    fator_neoplasias_final     = risk_radio("Outras neoplasias?")
    fator_menarca_final        = risk_radio("Menarca precoce?")

st.markdown("---")
# --- Limiar pré-definido ---
THRESHOLDS = {
    "0.3555 (Top 25%) – triagem ampla": 0.3555,
    "0.4814 (Top 10%) – equilíbrio prático": 0.4814,
    "0.5400 (Top 5%) – mais precisão": 0.5400,
    "0.6582 (Top 1%) – casos extremos": 0.6582,
}
choice = st.radio(
    "Escolha o limiar de decisão",
    list(THRESHOLDS.keys()),
    index=1,  # default: 0.4814
)
LIMIAR = THRESHOLDS[choice]

submit = st.button("Calcular probabilidade", type="primary", use_container_width=True)

# ------------------------- Inferência -------------------------
if submit:
    # monta vetor na ORDEM exata (float32)
    features = np.array([[
        float(gender),
        float(normalize_age(age_years)),
        float(fator_ca_ovario_final),
        float(fator_hf_mama_final),
        float(fator_genital_final),
        float(fator_obesidade_final),
        float(fator_sobrepeso_final),
        float(fator_etilismo_final),
        float(fator_tabagismo_final),
        float(fator_alimentacao_final),
        float(fator_sedentarismo_final),
        float(fator_radiacao_final),
        float(fator_anticoncepcao_final),
        float(fator_dislipidemia_final),
        float(fator_diabetes_final),
        float(fator_hipertensao_final),
        float(fator_gestacao_final),
        float(fator_mental_final),
        float(fator_neoplasias_final),
        float(fator_menarca_final),
    ]], dtype=np.float32)

    # sanity check
    n_expected = INPUT_SHAPE[-1] if isinstance(INPUT_SHAPE, (list, tuple)) else 20
    if features.shape[1] != n_expected:
        st.error(f"Ordem/quantidade de features divergente ({features.shape[1]} != {n_expected}).")
        st.stop()

    # inferência – procurar saída de probabilidades
    outs = sess.run(OUTPUT_NAMES, {INPUT_NAME: features})

    prob_pos = None; prob_neg = None

    if "probabilities" in OUTPUT_NAMES:
        p = outs[OUTPUT_NAMES.index("probabilities")]
        # pode vir (1,2) numpy, ou lista de dicionários
        try:
            parr = np.array(p)
            if parr.ndim == 2 and parr.shape[1] == 2:
                prob_neg, prob_pos = float(parr[0, 0]), float(parr[0, 1])
            elif parr.ndim == 2 and parr.shape[1] == 1:
                prob_pos = float(parr[0, 0]); prob_neg = 1.0 - prob_pos
        except Exception:
            pass
        if prob_pos is None and isinstance(p, list) and len(p) and isinstance(p[0], dict):
            d = p[0]
            prob_pos = float(d.get(1, d.get("1", 0.0)))
            prob_neg = float(d.get(0, d.get("0", 1.0)))

    # fallback: só label disponível
    if prob_pos is None:
        lbl = outs[OUTPUT_NAMES.index("label")] if "label" in OUTPUT_NAMES else outs[0]
        cls = int(np.array(lbl).ravel()[0])
        prob_pos = 1.0 if cls == 1 else 0.0
        prob_neg = 1.0 - prob_pos

    # ------------------- Exibição -------------------
    st.subheader("Resultado")
    classe = "Alto risco de lesão" if prob_pos >= LIMIAR else "Baixo risco de lesão"

    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Classe prevista", classe, help=f"Limiar atual: {LIMIAR:.4f}")
        st.write(f"**Prob. Lesão (classe 1):** {prob_pos:.1%}")
        st.write(f"**Prob. Não-Lesão (classe 0):** {prob_neg:.1%}")

    with colB:
        fig = go.Figure()

        # cores: verde p/ Não-Lesão, vermelho p/ Lesão
        colors = ["#2ecc71", "#e74c3c"]  # [não-lesão, lesão]

        fig.add_trace(go.Bar(
            x=["Não-Lesão", "Lesão"],
            y=[prob_neg, prob_pos],
            marker_color=colors,
            text=[f"{prob_neg:.1%}", f"{prob_pos:.1%}"],
            textposition="auto"
        ))

        # linha de limiar
        fig.add_hline(
            y=LIMIAR, line_dash="dash", line_color="#222",
            annotation_text=f"Limiar {LIMIAR:.4f}",
            annotation_position="top left"
        )

        fig.update_layout(
            yaxis=dict(range=[0, 1], title="Probabilidade"),
            xaxis=dict(title="Classe"),
            bargap=0.25,
            plot_bgcolor="rgba(0,0,0,0)",
            height=380
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Dados enviados", expanded=False):
            st.json({
                "gender": int(gender),
                "years_age (norm 0–1)": round(normalize_age(age_years), 3),
                **{k: int(v) for k, v in zip(ORDERED_FEATURES[2:], features[0, 2:].astype(int).tolist())}
            })

        st.caption("Aviso: ferramenta de apoio à decisão; não substitui julgamento clínico.")
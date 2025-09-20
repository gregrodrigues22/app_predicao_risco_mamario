# pages/explain.py
import os
from pathlib import Path
import streamlit as st

# =======================
# CONFIG
# =======================
st.set_page_config(page_title="🧠 Explicação do Modelo", layout="wide")
ASSETS = Path(__file__).resolve().parents[1] / "assets"

def first_existing(*names):
    for n in names:
        p = ASSETS / n
        if p.exists():
            return str(p)
    return None

LOGO = first_existing("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")

# =======================
# CABEÇALHO
# =======================
st.markdown(
    """
    <div style='background: linear-gradient(to right, #004e92, #000428);
                padding: 40px; border-radius: 12px; margin-bottom:30px'>
        <h1 style='color: white; margin: 0;'>🧠 Explicação do Modelo – Lesão Mamária</h1>
        <p style='color: #e8eef7; font-size:16px; margin-top:8px;'>
            Como o modelo foi construído, como performa e como interpretar suas saídas.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Oculta a navegação padrão do Streamlit na sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# MENU LATERAL
# =======================
with st.sidebar:
    if LOGO:
        st.image(LOGO, use_container_width=True)
    st.markdown("<hr style='border:none;border-top:1px solid #ccc;'/>", unsafe_allow_html=True)
    st.header("Menu")

    with st.expander("Predição no PoC", expanded=True):
        st.page_link("app.py", label="Predição no PoC", icon="📈")

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
        - 👥💬 Comunidade WhatsApp: patients2python
        """,
        unsafe_allow_html=True
    )

# =======================
# CONTEXTO E OBJETIVO (do PDF)
# =======================
st.subheader("Contexto clínico e objetivo")
st.markdown(
    """
O câncer de mama é a principal causa de mortalidade entre mulheres no Brasil. 
Com dados clínicos/administrativos e questionários, foi desenvolvido um **algoritmo de predição e estratificação de risco** para **problemas mamários** (benignos ou malignos) visando apoiar **APS** em prevenção, encaminhamento e diagnóstico precoce.
"""
)

# =======================
# DADOS & FEATURES (do PDF)
# =======================
st.subheader("Fontes de dados e engenharia de atributos")
st.markdown(
    """
**Bases utilizadas**: prontuário eletrônico (CID/CIAP/mensurações/atendimentos), contas médicas (CID/TUSS), 
questionários autoaplicáveis e logs de academias.  
A partir delas foram criadas **21 features booleanas** de fatores de risco (0/1) + **idade** normalizada (0–1).  
O **target** utilizado na modelagem foi “qualquer problema de mama” (benigno **ou** maligno) devido à baixa
ocorrência de malignidade isolada. :contentReference[oaicite:2]{index=2}
"""
)

with st.expander("Exemplos de fatores de risco considerados"):
    st.markdown(
        """
- História familiar/pessoal de CA de mama; neoplasia de ovário; patologia genital  
- Obesidade, sobrepeso, tabagismo, etilismo, sedentarismo, alimentação inadequada  
- Exposição à radiação, uso de anticoncepcional  
- Dislipidemia, diabetes, hipertensão, saúde mental  
- Outras neoplasias; menarca precoce  
        """
    )

# =======================
# TREINOS & MÉTRICA (do PDF)
# =======================
st.subheader("Modelagem e escolha de métrica")
st.markdown(
    """
Modelos comparados: **Logistic Regression**, **RandomForest**, **SVM**, **GaussianNB**, **KNN** e **Rede Neural (Sequential)**,
com **validação cruzada** e **otimização de hiperparâmetros**.  
Para comparação entre estudos e por cobrir toda a faixa de operação, a métrica primária reportada foi **AUC-ROC**.  
O melhor desempenho médio foi da **Regressão Logística** (``class_weight='balanced', max_iter=1000, penalty='l2'``),
com **AUC média = 0.8165** (IC95%: **0.8056 – 0.8229**).
"""
)

# =======================
# FIGURAS (como IMAGENS)
# Coloque os arquivos abaixo em assets/ para aparecerem:
#   - roc_curve.png
#   - pr_curve.png
#   - confusion_matrix.png
#   - hp_grid.png   (hiperparâmetros testados, opcional)
# =======================
st.subheader("Desempenho visual (imagens)")
roc_img = first_existing("roc_curve.png", "curva_roc.png")
pr_img  = first_existing("pr_curve.png", "precision_recall.png", "curva_pr.png")
cm_img  = first_existing("confusion_matrix.png", "matriz_confusao.png")
hp_img  = first_existing("hp_grid.png", "hiperparametros_testados.png")

cols = st.columns(2)
with cols[0]:
    if roc_img:
        st.image(roc_img, caption="Curva ROC (AUC ~ 0.82)", use_container_width=True)
    else:
        st.info("Adicione a imagem da Curva ROC em assets/ (ex.: roc_curve.png).")
with cols[1]:
    if pr_img:
        st.image(pr_img, caption="Curva Precision-Recall", use_container_width=True)
    else:
        st.info("Adicione a imagem da Curva Precision-Recall em assets/ (ex.: pr_curve.png).")

cols2 = st.columns(2)
with cols2[0]:
    if cm_img:
        st.image(cm_img, caption="Matriz de confusão (exemplo do melhor treino)", use_container_width=True)
    else:
        st.info("Adicione a imagem da Matriz de Confusão em assets/ (ex.: confusion_matrix.png).")

# =======================
# INTERPRETAÇÃO & USO
# =======================
st.subheader("Como interpretar e aplicar")
st.markdown(
    """
- **Probabilidade**: saída usada no app principal; os **limiares** sugeridos (25%, 10%, 5%, 1%) permitem
equilibrar **sensibilidade x precisão** conforme a **capacidade assistencial**.  
- **Triagem**: valores acima do limiar indicam priorização para **investigação/encaminhamento** na APS.  
- **Cuidado**: sempre associar o resultado a avaliação clínica, preferências do paciente/família e diretrizes.  
"""
)

st.info("Ferramenta de **apoio** à decisão — não substitui o julgamento clínico individual.")

# =======================
# LIMITAÇÕES & PRÓXIMOS PASSOS (do PDF)
# =======================
st.subheader("Limitações e próximos passos")
st.markdown(
    """
- **Rótulo**: para aumentar casos positivos, foi usada a junção **benigno + maligno** (reduz especificidade).  
- **Centro/dados**: dados operadora-específicos; recomenda-se **validação externa** e **recalibração** periódica.  
- **Custo-efetividade**: há proposta de **métrica customizada** que pondera custos de VP/VN/FP/FN para escolher o
**ponto de operação ideal** ao contexto (vide documento).
"""
)

st.caption("Baseado no relatório técnico do algoritmo de mamografia/lesão mamária do projeto (resumo e imagens).")

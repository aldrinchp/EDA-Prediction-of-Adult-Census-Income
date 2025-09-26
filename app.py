# app.py
# -----------------------------------------------------------
# Dashboard interactivo para Adult Income (ACI) con Streamlit
# Usa el dataset ya "preparado" desde eda.py (sin normalizar ni OHE)
# Usa despues el dataset preparado desde transformes.py (normalizado y OHE)
# -----------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import joblib

from eda import ACI_prepared  # noqa: F401


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = ACI_prepared.copy()
    # Etiqueta humana para income (0/1)
    if "income" in df.columns and df["income"].dtype != "object":
        df["income_label"] = df["income"].map({0: "<=50K", 1: ">50K"}).astype("category")
    return df

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_ACI.pkl")
model = load_model()

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)

def quantile_bounds(s: pd.Series, qlow=0.01, qhigh=0.99):
    # Evita rangos extremos por outliers
    s_num = pd.to_numeric(s, errors="coerce")
    return float(s_num.quantile(qlow)), float(s_num.quantile(qhigh))

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Adult Income – EDA interactivo + Predicción",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Income (ACI): Exploración Interactiva")
st.caption("Gráficas interactivas del Dataset Adult Census Income (ACI). Además, predicción de ingreso (>50K) con un modelo de Árbol de Decisión.")

# Cargar datos
df = get_data()

# Panel superior: métricas rápidas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{len(df):,}")
c2.metric("Columnas", f"{df.shape[1]:,}")
missing_pct = df.isna().mean().mean() * 100
c3.metric("% faltantes (prom.)", f"{missing_pct:.2f}%")
if "income" in df.columns:
    p50k = df["income"].mean() * 100 if is_numeric(df["income"]) else df["income_label"].eq(">50K").mean() * 100
    c4.metric("% clase >50K", f"{p50k:.1f}%")
else:
    c4.metric("% clase >50K", "—")

st.markdown("---")

# -----------------------------
# Sidebar: Filtros dinámicos
# -----------------------------
st.sidebar.header("Filtros")
st.sidebar.caption("Ajusta los filtros para segmentar el dataset.")

numeric_cols = [c for c in df.columns if is_numeric(df[c])]
categorical_cols = [c for c in df.columns if is_categorical(df[c])]

# Filtros numéricos
with st.sidebar.expander("Variables numéricas", expanded=False):
    num_filters = {}
    for col in numeric_cols:
        s = df[col]
        if s.notna().sum() == 0:
            continue
        lo, hi = quantile_bounds(s)
        min_v = float(np.nanmin(s))
        max_v = float(np.nanmax(s))
        # Asegura rangos crecientes
        lo = max(min_v, lo)
        hi = min(max_v, hi)
        fmin, fmax = st.slider(
            f"{col}",
            min_value=min_v, max_value=max_v,
            value=(lo, hi),
            step=(max_v - min_v) / 100 if np.isfinite(max_v - min_v) and (max_v - min_v) > 0 else 1.0
        )
        num_filters[col] = (fmin, fmax)

# Filtros categóricos
with st.sidebar.expander("Variables categóricas", expanded=False):
    cat_filters = {}
    for col in categorical_cols:
        opts = sorted([str(x) for x in df[col].dropna().unique().tolist()])
        if len(opts) == 0:
            continue
        selected = st.multiselect(f"{col}", options=opts, default=[])
        cat_filters[col] = selected

# Aplicar filtros
mask = pd.Series(True, index=df.index)
# Numéricos
for col, (lo, hi) in num_filters.items():
    if col in df.columns:
        mask &= df[col].between(lo, hi)
# Categóricos
for col, selected in cat_filters.items():
    if selected:
        mask &= df[col].astype(str).isin(selected)

df_f = df[mask].copy()

st.subheader("Vista de datos filtrados")
st.caption("Descarga el subconjunto filtrado o explóralo abajo.")
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV filtrado", data=csv, file_name="adult_filtered.csv", mime="text/csv")

st.dataframe(df_f, use_container_width=True, height=420)

st.markdown("---")

# -----------------------------
# Sección: Histogramas / Barras
# -----------------------------
st.header("Distribuciones")
colA, colB = st.columns([2, 1])

with colA:
    hist_col = st.selectbox(
        "Elige la variable para el histograma / barras",
        options=[*numeric_cols, *categorical_cols],
        index=0 if numeric_cols else 0
    )

with colB:
    hue_col = st.selectbox(
        "Color por (opcional)",
        options=["(ninguno)"] + [c for c in df_f.columns if c != hist_col],
        index=0
    )

if hist_col:
    if is_numeric(df_f[hist_col]):
        fig_hist = px.histogram(
            df_f, x=hist_col,
            color=None if hue_col == "(ninguno)" else hue_col,
            nbins=40,
            marginal="box",  # agrega boxplot superior
            opacity=0.9
        )
        fig_hist.update_layout(height=450, bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        fig_bar = px.histogram(
            df_f, x=hist_col,
            color=None if hue_col == "(ninguno)" else hue_col,
            histfunc="count"
        )
        fig_bar.update_layout(height=450, bargap=0.2)
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# -----------------------------
# Sección: Relaciones (Scatter / Box)
# -----------------------------
st.header("Relaciones entre variables")

t1, t2, t3 = st.tabs(["Dispersión (num vs num)", "Box/Violin (num vs cat)", "Tabla dinámica"])

with t1:
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("Eje X (num)", options=numeric_cols, index=0 if numeric_cols else None)
    y_col = c2.selectbox("Eje Y (num)", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    color_col = c3.selectbox("Color por", options=["(ninguno)"] + df_f.columns.tolist(), index=0)
    if x_col and y_col:
        fig_sc = px.scatter(
            df_f, x=x_col, y=y_col,
            color=None if color_col == "(ninguno)" else color_col,
            trendline="ols"
        )
        fig_sc.update_layout(height=500)
        st.plotly_chart(fig_sc, use_container_width=True)

with t2:
    c1, c2 = st.columns(2)
    num_for_box = c1.selectbox("Variable numérica", options=numeric_cols, index=0 if numeric_cols else None, key="box_num")
    cat_for_box = c2.selectbox("Variable categórica", options=categorical_cols, index=0 if categorical_cols else None, key="box_cat")
    if num_for_box and cat_for_box:
        fig_box = px.box(df_f, x=cat_for_box, y=num_for_box, points="all")
        fig_box.update_layout(height=520, xaxis_tickangle=-30)
        st.plotly_chart(fig_box, use_container_width=True)

with t3:
    st.dataframe(df_f.groupby(categorical_cols[:1] or ["income_label"]).agg({c: ["mean", "median"] for c in numeric_cols}), use_container_width=True)

st.markdown("---")

# -----------------------------
# Sección: Correlación
# -----------------------------
st.header("Matriz de correlación (numéricas)")
st.caption("Selecciona método de correlación (Pearson, Spearman o Kendall). Solo usa columnas numéricas.")
corr_method = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True, index=0)

num_for_corr = df_f[numeric_cols].copy()
if num_for_corr.shape[1] >= 2:
    corr = num_for_corr.corr(method=corr_method)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower",
        zmin=-1, zmax=1,
        labels=dict(color="corr")
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 columnas numéricas para calcular correlación.")



st.header("Predicción de ingreso (>50K)")

with st.expander("Haz una predicción con el modelo de Árbol de Decisión"):
    with st.form("prediction_form"):
        st.write("Introduce los valores de la persona:")

        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        workclass = st.selectbox("Workclass", sorted(df["workclass"].dropna().unique()))
        fnlwgt = st.number_input("Fnlwgt", min_value=0, max_value=int(df["fnlwgt"].max()), value=int(df["fnlwgt"].median()))
        education = st.selectbox("Educación", sorted(df["education"].dropna().unique()))
        education_num = st.number_input("Education-num", min_value=1, max_value=20, value=10)
        marital_status = st.selectbox("Estado civil", sorted(df["marital.status"].dropna().unique()))
        occupation = st.selectbox("Ocupación", sorted(df["occupation"].dropna().unique()))
        relationship = st.selectbox("Relación", sorted(df["relationship"].dropna().unique()))
        race = st.selectbox("Raza", sorted(df["race"].dropna().unique()))
        sex = st.selectbox("Sexo", sorted(df["sex"].dropna().unique()))
        capital_gain = st.number_input("Capital-gain", min_value=0, max_value=int(df["capital.gain"].max()), value=0)
        capital_loss = st.number_input("Capital-loss", min_value=0, max_value=int(df["capital.loss"].max()), value=0)
        hours_per_week = st.number_input("Horas por semana", min_value=1, max_value=100, value=40)
        native_country = st.selectbox("País de origen", sorted(df["native.country"].dropna().unique()))

        submitted = st.form_submit_button("Predecir")

    if submitted:
        # Construir el DataFrame de entrada
        input_dict = {
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [fnlwgt],
            'education': [education],
            'education.num': [education_num],
            'marital.status': [marital_status],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'sex': [sex],
            'capital.gain': [capital_gain],
            'capital.loss': [capital_loss],
            'hours.per.week': [hours_per_week],
            'native.country': [native_country]
        }
        input_df = pd.DataFrame(input_dict)

        # Preprocesar igual que en entrenamiento
        from transformes import data_preparer  
        input_prep = data_preparer.transform(input_df)

        # Predecir
        pred = model.predict(input_prep)[0]
        pred_label = ">50K" if pred == 1 else "<=50K"
        st.success(f"Predicción: {pred_label}")

st.markdown("---")

# -----------------------------
# Info del dataset
# -----------------------------
with st.expander("Información de columnas y tipos"):
    dtypes = pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str)})
    st.dataframe(dtypes, use_container_width=True, height=260)


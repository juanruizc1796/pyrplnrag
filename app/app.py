import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================
# 0. CONFIGURACIÓN
# ============================

ARTICLES_PATH = "data/processed/articulos_total.csv"
EMBEDDINGS_PATH = "data/processed/models/embeddings_total.npy"
FAISS_PATH = "data/processed/models/faiss_index_total.bin"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


# ============================
# 1. CARGA DE MODELOS Y DATOS
# ============================

@st.cache_resource
def load_resources():
    df = pd.read_csv(ARTICLES_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(FAISS_PATH)

    encoder = SentenceTransformer(EMBED_MODEL_NAME)

    groq_api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", None))
    if groq_api_key is None:
        raise ValueError("No se encontró GROQ_API_KEY en variables de entorno ni en st.secrets.")

    groq_client = Groq(api_key=groq_api_key)

    return df, embeddings, index, encoder, groq_client


df, embeddings, index, encoder, groq_client = load_resources()


# ============================
# 2. RAG
# ============================

def buscar_articulos(query, top_k=4):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb).astype("float32"), top_k)

    resultados = df.iloc[idxs[0]].copy()
    resultados["score"] = scores[0]
    return resultados


def call_groq(prompt: str) -> str:

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en normativa de tránsito en Colombia. "
                    "Responde siempre basándote EXCLUSIVAMENTE en el contexto proporcionado. "
                    "Si la respuesta no está en el contexto, responde: "
                    "'No encontré información relevante en la normativa cargada.' "
                    "No inventes leyes. Usa lenguaje claro y pedagógico."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return completion.choices[0].message.content


def rag_responder(query: str, top_k: int = 4):

    articulos = buscar_articulos(query, top_k=top_k)

    contexto = ""
    for _, row in articulos.iterrows():
        contexto += f"ARTÍCULO {row['id_articulo']} - {row['titulo']}\n{row['texto']}\n\n"

    prompt = f"""
PREGUNTA:
{query}

ARTÍCULOS RELEVANTES (EXTRACTOS DE NORMATIVA):
{contexto}

Instrucciones:
- Responde de manera clara y pedagógica.
- Indica explícitamente qué artículos usas.
- Si no hay información suficiente, respóndelo.
"""

    respuesta = call_groq(prompt)
    return respuesta



# ============================
# 3. INTERFAZ STREAMLIT
# ============================

st.set_page_config(page_title="Asistente Legal de Tránsito", page_icon="🚦")

st.title("🚦 Asistente Legal de Tránsito • Colombia")
st.write("Consulta normativa oficial usando un sistema RAG (búsqueda + LLM).")

st.subheader("Preguntas sugeridas:")
col1, col2, col3 = st.columns(3)

q1 = "¿Qué debo hacer si me imponen un comparendo?"
q2 = "¿Cuánto tiempo tengo para pagar una multa?"
q3 = "¿En qué casos inmovilizan mi vehículo?"

if col1.button(q1):
    st.session_state["pregunta"] = q1

if col2.button(q2):
    st.session_state["pregunta"] = q2

if col3.button(q3):
    st.session_state["pregunta"] = q3


pregunta = st.text_input(
    "Pregunta sobre tránsito, multas, licencias, señalización, etc:",
    value=st.session_state.get("pregunta", "")
)


if st.button("Consultar"):
    if pregunta.strip() == "":
        st.warning("Por favor ingresa una pregunta.")
    else:
        with st.spinner("Generando respuesta..."):
            respuesta = rag_responder(pregunta)

        st.subheader("📌 Respuesta")
        st.write(respuesta)

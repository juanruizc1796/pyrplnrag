import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from groq import Groq  # Cliente oficial de Groq


# ============================
# 0. CONFIGURACIÓN
# ============================

ARTICLES_PATH = "data/processed/articulos_total.csv"
EMBEDDINGS_PATH = "data/processed/models/embeddings_total.npy"
FAISS_PATH = "data/processed/models/faiss_index_total.bin"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Ajusta al nombre exacto de tu modelo en Groq


# ============================
# 1. CARGA DE MODELOS Y DATOS
# ============================

@st.cache_resource
def load_resources():
    # Cargar datos
    df = pd.read_csv(ARTICLES_PATH)

    # Cargar embeddings (aunque acá no los usamos directamente, solo el índice)
    embeddings = np.load(EMBEDDINGS_PATH)

    # Cargar índice FAISS
    index = faiss.read_index(FAISS_PATH)

    # Cargar modelo de embeddings
    encoder = SentenceTransformer(EMBED_MODEL_NAME)

    # Cliente Groq (usa la API key desde secrets o variables de entorno)
    groq_api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", None))
    if groq_api_key is None:
        raise ValueError("No se encontró GROQ_API_KEY en variables de entorno ni en st.secrets.")
    groq_client = Groq(api_key=groq_api_key)

    return df, embeddings, index, encoder, groq_client


df, embeddings, index, encoder, groq_client = load_resources()


# ============================
# 2. RAG SIMPLE
# ============================

def buscar_articulos(query, top_k=3):
    """
    Busca los top_k artículos más relevantes usando FAISS + embeddings.
    """
    q_emb = encoder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb).astype("float32"), top_k)
    resultados = df.iloc[idxs[0]].copy()
    resultados["score"] = scores[0]
    return resultados


def call_groq(prompt: str) -> str:
    """
    Llama al modelo de Groq con el prompt dado.
    """
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en normativa de tránsito en Colombia. "
                    "Responde siempre citando los artículos relevantes del contexto que se te entrega. "
                    "No inventes información que no esté en el contexto y explica en lenguaje claro."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return completion.choices[0].message.content


def rag_responder(query: str, top_k: int = 3):
    """
    RAG simple: recupera artículos y genera respuesta usando Groq.
    """
    articulos = buscar_articulos(query, top_k=top_k)

    contexto = ""
    for _, row in articulos.iterrows():
        contexto += f"ARTÍCULO {row['id_articulo']} - {row['titulo']}\n{row['texto_articulo']}\n\n"

    prompt = f"""
PREGUNTA:
{query}

ARTÍCULOS RELEVANTES (EXTRACTOS DE NORMATIVA):
{contexto}

Instrucciones:
- Responde de manera clara y pedagógica.
- Indica explícitamente qué artículos usas (por número).
- Si la respuesta no puede darse con este contexto, dilo honestamente.
"""

    respuesta = call_groq(prompt)

    return respuesta, articulos


# ============================
# 3. INTERFAZ STREAMLIT
# ============================

st.set_page_config(page_title="Asistente Legal de Tránsito", page_icon="🚦")

st.title("🚦 Asistente Legal de Tránsito • Colombia")
st.write(
    "Consulta normativa de tránsito basada en un sistema RAG que combina la Ley 769 y otras normas complementarias."
)

pregunta = st.text_input("Ingresa tu pregunta sobre tránsito, multas, licencias, señalización, etc:")

top_k = st.slider("Número de artículos a considerar (top_k)", min_value=1, max_value=10, value=3)

if st.button("Consultar"):
    if pregunta.strip() == "":
        st.warning("Por favor ingresa una pregunta.")
    else:
        with st.spinner("Generando respuesta..."):
            respuesta, arts = rag_responder(pregunta, top_k=top_k)

        st.subheader("📌 Respuesta")
        st.write(respuesta)

        st.subheader("📚 Artículos usados")
        st.dataframe(arts[["id_articulo", "titulo", "score"]])

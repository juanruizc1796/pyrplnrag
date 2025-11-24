import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

df = pd.read_csv("data/processed/articulos_total.csv")
embeddings = np.load("data/processed/models/embeddings_total.npy")
index = faiss.read_index("data/processed/models/faiss_index_total.bin")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
llm = OllamaLLM(model="llama3.1")

def buscar_articulos(query, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb).astype("float32"), top_k)

    resultados = df.iloc[idxs[0]].copy()
    resultados["score"] = scores[0]
    resultados = resultados.sort_values(by="score", ascending=False)
    return resultados

def rag_responder(query, top_k=3):

    articulos = buscar_articulos(query, top_k)

    contexto = ""
    for _, row in articulos.iterrows():
        contexto += f"ARTÍCULO {row['id_articulo']} - {row['titulo']}\n{row['texto_articulo']}\n\n"

    prompt = f"""
Eres un asistente experto en normativa de tránsito en Colombia.
Responde de manera clara, pedagógica y cita siempre los artículos usados.

PREGUNTA:
{query}

ARTÍCULOS RELEVANTES:
{contexto}

RESPUESTA:
"""

    respuesta = llm.invoke(prompt)

    return respuesta, articulos


if __name__ == "__main__":
    pregunta = "¿En qué casos me pueden cancelar la licencia?"
    respuesta, articulos = rag_responder(pregunta)

    print("\n=== RESPUESTA ===\n")
    print(respuesta)
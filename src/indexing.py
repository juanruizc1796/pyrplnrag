import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

DATA_PROCESSED = Path("data/processed")
MODELS_DIR = DATA_PROCESSED / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_PATH = DATA_PROCESSED / "articulos_total.csv"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings_total.npy"
INDEX_PATH = MODELS_DIR / "faiss_index_total.bin"


def load_articles():
    df = pd.read_csv(ARTICLES_PATH)

    df = df.dropna(subset=["texto_articulo"])
    df = df[df["texto_articulo"].str.strip() != ""]
    df["texto_articulo"] = df["texto_articulo"].astype(str)

    print(f"Artículos cargados: {len(df)}")
    return df


def build_embeddings(df):
    print("Cargando modelo de embeddings (MiniLM-L12)...\n")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    textos = df["texto_articulo"].tolist()

    textos = [str(t) for t in textos]

    embeddings = model.encode(
        textos,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"✔ Embeddings guardados en {EMBEDDINGS_PATH}")

    return embeddings


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    print("\nConstruyendo índice FAISS...")
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Índice FAISS guardado en {INDEX_PATH}")

    return index


def main():
    print("Construcción de embeddings + índice FAISS ====\n")

    df = load_articles()
    embeddings = build_embeddings(df)
    build_faiss_index(embeddings)

    print("\nProceso completado: Embeddings + Índice FAISS listo.")


if __name__ == "__main__":
    main()

import re
import pdfplumber
import pandas as pd
from pathlib import Path

# ========================================
# CONFIG
# ========================================
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 400          # tamaño aproximado de chunk (en tokens aprox)
CHUNK_OVERLAP = 0.2       # 20% de solapamiento
MIN_CHARS = 200           # longitud mínima para aceptar un chunk


# ========================================
# 1. EXTRAER TEXTO DE PDF
# ========================================
def extract_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
    return "\n".join(pages)


# ========================================
# 2. NORMALIZACIÓN DE TEXTO
# ========================================
def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


# ========================================
# 3. CHUNKING UNIVERSAL POR VENTANA
# ========================================
def chunk_by_window(text: str,
                    chunk_size: int = CHUNK_SIZE,
                    overlap_ratio: float = CHUNK_OVERLAP) -> pd.DataFrame:

    words = text.split()
    total_words = len(words)

    step = int(chunk_size * (1 - overlap_ratio))

    chunks = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        # evitar chunks muy pequeños
        if len(chunk) > MIN_CHARS:
            chunks.append(chunk)

        start += step

    df = pd.DataFrame({
        "id_articulo": [None] * len(chunks),
        "titulo": [""] * len(chunks),
        "texto": chunks,
        "tipo": ["ventana"] * len(chunks)
    })

    return df


# ========================================
# 4. MAIN LOOP
# ========================================
def main():
    all_rows = []

    for pdf_path in DATA_RAW.glob("*.pdf"):
        print(f"Procesando: {pdf_path.name}")

        raw = extract_text(pdf_path)
        raw = clean_text(raw)

        df = chunk_by_window(raw)

        df["fuente_pdf"] = pdf_path.name
        all_rows.append(df)

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(DATA_PROCESSED / "articulos_total.csv", index=False)

    print("=====================================")
    print(f"✔ Total chunks generados: {len(final_df)}")
    print("✔ Guardado en data/processed/articulos_total.csv")
    print("=====================================")


if __name__ == "__main__":
    main()

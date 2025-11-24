import re
import pdfplumber
import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ===================================================
# 1. EXTRAER TEXTO DE PDF
# ===================================================
def extract_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
    return "\n".join(pages)

# ===================================================
# 2A. CHUNKS PARA LEYES (ARTÍCULOS)
# ===================================================
def chunk_by_articles(text: str) -> pd.DataFrame:

    # Detecta: ARTÍCULO, Artículo, ARTICULO, articulo etc.
    pattern = re.compile(
        r"(ART[ÍI]CULO\s+\d+\.?(?:\s+[^\n]+)?)",
        flags=re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    rows = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        header = m.group(1).strip()
        chunk = text[start:end].strip()

        # extrae id
        id_match = re.search(r"ART[ÍI]CULO\s+(\d+)", header, flags=re.IGNORECASE)
        id_art = int(id_match.group(1)) if id_match else None

        # título si existe
        title_match = re.split(r"ART[ÍI]CULO\s+\d+\.?", header, flags=re.IGNORECASE)
        titulo = title_match[1].strip() if len(title_match) > 1 else ""

        # remover header del texto
        lines = chunk.split("\n")
        body = "\n".join(lines[1:]).strip()

        rows.append({
            "id_articulo": id_art,
            "titulo": titulo,
            "texto": body,
            "tipo": "articulo"
        })

    return pd.DataFrame(rows)

# ===================================================
# 2B. CHUNKS PARA MANUAL (1.x, 2.x, 3.x…)
# ===================================================
def chunk_by_manual_sections(text: str) -> pd.DataFrame:
    """
    Chunking por secciones principales del manual:
    1.
    1.x
    2.
    2.x
    etc.
    """

    # detecta capítulos y secciones principales
    pattern = re.compile(
        r"(CAP[ÍI]TULO\s+\d+|^\d+\.\s+[^\n]+|^\d+\.\d+)",
        flags=re.IGNORECASE | re.MULTILINE
    )

    matches = list(pattern.finditer(text))
    rows = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        header = m.group(1).strip()
        chunk = text[start:end].strip()

        rows.append({
            "id_articulo": None,
            "titulo": header,
            "texto": chunk,
            "tipo": "manual"
        })

    return pd.DataFrame(rows)

# ===================================================
# 3. MAIN LOOP
# ===================================================
def main():
    all_rows = []

    for pdf_path in DATA_RAW.glob("*.pdf"):
        print(f"Procesando: {pdf_path.name}")

        raw = extract_text(pdf_path)

        # limpieza leve
        raw = raw.replace("\r", "\n")
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{2,}", "\n\n", raw)

        # REGLA: si es el manual, usar chunking especial
        if "manual" in pdf_path.name.lower():
            df = chunk_by_manual_sections(raw)
        else:
            df = chunk_by_articles(raw)

        df["fuente_pdf"] = pdf_path.name
        all_rows.append(df)

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(DATA_PROCESSED / "articulos_total.csv", index=False)

    print(f"✔ Total chunks generados: {len(final_df)}")
    print(f"✔ Guardado en data/processed/articulos_total.csv")


if __name__ == "__main__":
    main()

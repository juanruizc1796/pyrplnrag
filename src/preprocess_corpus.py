import re
import pdfplumber
import pandas as pd
from pathlib import Path


DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


article_pattern = re.compile(
    r"(ART[ÍI]CULO\s+\d+\.?(?:\s+[^\n]+)?)",
    flags=re.IGNORECASE
)


def extract_text(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


def clean_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t


def split_articles(text: str, source: str) -> pd.DataFrame:
    matches = list(article_pattern.finditer(text))
    rows = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        header = m.group(1)
        num = re.search(r"ART[ÍI]CULO\s+(\d+)", header, re.I)
        id_art = int(num.group(1)) if num else None

        title = re.split(r"ART[ÍI]CULO\s+\d+\.?", header, flags=re.I)
        title = title[1].strip() if len(title) > 1 else ""

        content = text[start:end]
        lines = content.split("\n")
        body = "\n".join(lines[1:]).strip()

        rows.append({
            "fuente": source,
            "id_articulo": id_art,
            "titulo": title,
            "texto_articulo": body
        })
    return pd.DataFrame(rows)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for pdf_path in DATA_RAW.glob("*.pdf"):
        print(f"Procesando: {pdf_path.name}")
        raw = extract_text(pdf_path)
        clean = clean_text(raw)
        df = split_articles(clean, source=pdf_path.name)
        all_rows.append(df)

    final_df = pd.concat(all_rows, ignore_index=True)

    final_df.to_csv(DATA_PROCESSED / "articulos_total.csv", index=False)
    print(f"✔ Artículos procesados: {len(final_df)}")
    print("✔ Guardado en data/processed/articulos_total.csv")


if __name__ == "__main__":
    main()

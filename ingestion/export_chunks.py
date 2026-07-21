import json
from dataclasses import asdict
from pathlib import Path

from ingestion.chunker import chunk_sections
from ingestion.loader import load_docx_sections


def main() -> None:
    source_path = Path("docs/packet_core/TS23501.docx")
    output_path = Path("data/processed/ts23501_chunks.json")

    sections = load_docx_sections(source_path)
    chunks = chunk_sections(sections, max_chars=1200)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [asdict(chunk) for chunk in chunks],
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Sections loaded: {len(sections)}")
    print(f"Chunks exported: {len(chunks)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
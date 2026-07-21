from dataclasses import dataclass
from typing import List

from ingestion.loader import SpecSection


@dataclass
class DocumentChunk:
    spec: str
    release: str
    version: str
    section: str
    title: str
    chunk_id: int
    text: str
    source: str


def chunk_sections(
    sections: List[SpecSection],
    max_chars: int = 1200,
) -> List[DocumentChunk]:
    chunks = []

    for section in sections:
        paragraphs = [
            paragraph.strip()
            for paragraph in section.content.split("\n")
            if paragraph.strip()
        ]

        current = ""
        chunk_id = 1

        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 1 <= max_chars:
                if current:
                    current += "\n"

                current += paragraph

            else:
                if current:
                    chunks.append(
                        DocumentChunk(
                            spec=section.spec,
                            release=section.release,
                            version=section.version,
                            section=section.section,
                            title=section.title,
                            chunk_id=chunk_id,
                            text=current,
                            source=section.source,
                        )
                    )

                    chunk_id += 1

                current = paragraph

        if current:
            chunks.append(
                DocumentChunk(
                    spec=section.spec,
                    release=section.release,
                    version=section.version,
                    section=section.section,
                    title=section.title,
                    chunk_id=chunk_id,
                    text=current,
                    source=section.source,
                )
            )

    return chunks


if __name__ == "__main__":
    from pathlib import Path

    from ingestion.loader import load_docx_sections

    document_path = Path("docs/packet_core/TS23501.docx")

    sections = load_docx_sections(document_path)

    chunks = chunk_sections(sections, max_chars=1200)

    print(f"Sections loaded: {len(sections)}")
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        sample = chunks[0]

        print("\nSample chunk:")
        print(f"Spec: {sample.spec}")
        print(f"Section: {sample.section}")
        print(f"Title: {sample.title}")
        print(f"Chunk ID: {sample.chunk_id}")
        print(f"Characters: {len(sample.text)}")
        print("-" * 60)
        print(sample.text[:500])
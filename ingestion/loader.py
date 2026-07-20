import re
from dataclasses import asdict, dataclass
from pathlib import Path

from docx import Document


SECTION_PATTERN = re.compile(
    r"^(?P<section>(?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)*))\s+(?P<title>.+)$"
)


@dataclass
class SpecSection:
    spec: str
    release: str
    version: str
    section: str
    title: str
    content: str
    source: str


def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_heading(paragraph) -> bool:
    style_name = paragraph.style.name if paragraph.style else ""
    return style_name.lower().startswith("heading")


def parse_heading(text: str) -> tuple[str, str] | None:
    match = SECTION_PATTERN.match(clean_text(text))

    if not match:
        return None

    return match.group("section"), match.group("title")


def load_docx_sections(
    file_path: str | Path,
    *,
    max_sections: int | None = None,
) -> list[SpecSection]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    document = Document(path)

    sections: list[SpecSection] = []
    current_section: str | None = None
    current_title: str | None = None
    current_content: list[str] = []

    spec = "TS 23.501"
    release = "19"
    version = "19.8.0"

    def save_current_section() -> None:
        nonlocal current_section, current_title, current_content

        if not current_section or not current_title:
            return

        content = "\n".join(current_content).strip()

        if content:
            sections.append(
                SpecSection(
                    spec=spec,
                    release=release,
                    version=version,
                    section=current_section,
                    title=current_title,
                    content=content,
                    source=path.name,
                )
            )

        current_content = []

    for paragraph in document.paragraphs:
        text = clean_text(paragraph.text)

        if not text:
            continue

        parsed_heading = parse_heading(text) if is_heading(paragraph) else None

        if parsed_heading:
            save_current_section()

            current_section, current_title = parsed_heading

            if max_sections and len(sections) >= max_sections:
                break
        elif current_section:
            current_content.append(text)

    if not max_sections or len(sections) < max_sections:
        save_current_section()

    return sections


def main() -> None:
    file_path = Path("docs/packet_core/TS23501.docx")

    sections = load_docx_sections(file_path, max_sections=20)

    print(f"Extracted {len(sections)} sections.\n")

    for section in sections[:10]:
        print(f"{section.section} — {section.title}")
        print(section.content[:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
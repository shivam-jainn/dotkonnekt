import fitz  # PyMuPDF

from src.core.document import (
    Block,
    Document,
    Heading,
    Image,
    Page,
    Paragraph,
)
from src.core.parsers.base import Parser, ParsedDocument


def _classify_block(text: str, font_size: float | None = None) -> tuple[str, int | None]:
    """Classify a text block into kind and optional heading level.

    Uses font size heuristics and structural patterns.
    """
    stripped = text.strip()
    if not stripped:
        return "paragraph", None

    # Heuristic: large font = heading
    if font_size is not None and font_size >= 16:
        level = 1 if font_size >= 20 else 2 if font_size >= 16 else 3
        return "heading", level

    lines = stripped.split("\n")
    first_line = lines[0].strip()

    # Markdown-style headings
    if first_line.startswith("#"):
        hashes = len(first_line) - len(first_line.lstrip("#"))
        return "heading", min(hashes, 6)

    # ALL CAPS short line = heading
    if len(lines) <= 2 and first_line.isupper() and len(first_line) < 80:
        return "heading", 2

    # Legal section headings
    import re
    if re.match(
        r"^(?:ARTICLE|SECTION|CLAUSE|PART|CHAPTER)\s+\S+",
        first_line,
        re.IGNORECASE,
    ):
        return "heading", 2

    # Numbered section headings (e.g., "1. Introduction", "3.2 Scope")
    if re.match(r"^\d{1,3}(?:\.\d{1,3}){0,2}\.\s+\S", first_line):
        dots = first_line.split()[0].count(".")
        return "heading", min(dots, 3)

    return "paragraph", None


class PDFParser(Parser):
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        doc = fitz.open(stream=data, filetype="pdf")
        pages: list[str] = []
        extracted_images: list[dict] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append(text)

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    extracted_images.append(
                        {
                            "page": page_num,
                            "index": img_index,
                            "ext": base_image["ext"],
                            "bytes": base_image["image"],
                        }
                    )

        doc.close()

        return ParsedDocument(
            content="\n\n".join(pages),
            metadata={
                "filename": filename,
                "total_pages": len(pages),
                "parser": "pdf",
            },
            extracted_images=extracted_images,
        )

    def parse_to_document(self, data: bytes, filename: str) -> Document:
        """Parse PDF directly into Document IR with structured blocks."""
        doc = fitz.open(stream=data, filetype="pdf")
        document = Document(filename=filename)

        extracted_images: list[Image] = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Structured text extraction (blocks with position info)
            text_dict = page.get_text("dict")
            ir_page = Page(number=page_num + 1)

            raw_text_parts: list[str] = []

            for block in text_dict.get("blocks", []):
                if block["type"] == 0:  # text block
                    lines: list[str] = []
                    max_font_size: float = 0

                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                            font_size = span.get("size", 12)
                            if font_size > max_font_size:
                                max_font_size = font_size
                        lines.append(line_text)

                    block_text = "\n".join(lines).strip()
                    if not block_text:
                        continue

                    raw_text_parts.append(block_text)
                    kind, level = _classify_block(block_text, max_font_size)

                    ir_block = Block(
                        kind=kind,
                        content=block_text,
                        level=level,
                        page=page_num + 1,
                    )
                    ir_page.blocks.append(ir_block)

                    if kind == "heading":
                        ir_page.headings.append(
                            Heading(text=block_text, level=level or 1, page=page_num + 1)
                        )
                    else:
                        ir_page.paragraphs.append(
                            Paragraph(text=block_text, page=page_num + 1)
                        )

                elif block["type"] == 1:  # image block
                    img_data = block.get("image")
                    if img_data:
                        ir_image = Image(
                            bytes=img_data,
                            ext="png",
                            page=page_num + 1,
                            index=len(ir_page.images),
                        )
                        ir_page.images.append(ir_image)
                        extracted_images.append(ir_image)

            ir_page.text = "\n\n".join(raw_text_parts)
            document.pages.append(ir_page)

        document.raw_text = "\n\n".join(p.text for p in document.pages if p.text)
        document.metadata = {
            "filename": filename,
            "total_pages": len(document.pages),
            "parser": "pdf",
        }

        # Attach extracted images as dicts for backward compat with graph nodes
        document.metadata["extracted_images"] = [
            {"page": img.page, "index": img.index, "ext": img.ext, "bytes": img.bytes}
            for img in extracted_images
        ]

        doc.close()
        return document

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

import fitz  # PyMuPDF

from src.core.parsers.base import Parser, ParsedDocument


class PDFParser(Parser):
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        doc = fitz.open(stream=data, filetype="pdf")
        pages: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append(text)

        doc.close()

        return ParsedDocument(
            content="\n\n".join(pages),
            metadata={
                "filename": filename,
                "total_pages": len(pages),
                "parser": "pdf",
            },
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

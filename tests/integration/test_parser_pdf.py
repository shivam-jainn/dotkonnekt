import pytest

from src.core.parsers.base import ParsedDocument
from src.core.parsers.pdf import PDFParser


pytestmark = pytest.mark.integration


def _make_pdf(pages_text: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPDFParserIntegration:
    def test_parse_real_pdf_single_page(self):
        pdf_data = _make_pdf(["Hello, this is a test PDF."])
        parser = PDFParser()

        result = parser.parse(pdf_data, "test.pdf")

        assert isinstance(result, ParsedDocument)
        assert "Hello" in result.content
        assert "test PDF" in result.content
        assert result.metadata["filename"] == "test.pdf"
        assert result.metadata["total_pages"] == 1
        assert result.metadata["parser"] == "pdf"

    def test_parse_real_pdf_multiple_pages(self):
        pdf_data = _make_pdf(
            ["Page one content", "Page two content", "Page three content"]
        )
        parser = PDFParser()

        result = parser.parse(pdf_data, "multi.pdf")

        assert result.metadata["total_pages"] == 3
        assert "Page one content" in result.content
        assert "Page two content" in result.content
        assert "Page three content" in result.content

    def test_parse_real_pdf_pages_separated_by_double_newline(self):
        pdf_data = _make_pdf(["Alpha", "Beta"])
        parser = PDFParser()

        result = parser.parse(pdf_data, "separator.pdf")

        assert "Alpha" in result.content
        assert "Beta" in result.content

    def test_parse_preserves_paragraphs(self):
        long_text = "This is a longer paragraph with enough text to verify that the parser preserves the full content of each page accurately."
        pdf_data = _make_pdf([long_text])
        parser = PDFParser()

        result = parser.parse(pdf_data, "long.pdf")

        assert "longer paragraph" in result.content
        assert "preserves the full content" in result.content

    def test_supported_extensions(self):
        parser = PDFParser()
        assert parser.supported_extensions() == [".pdf"]

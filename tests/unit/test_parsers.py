from unittest.mock import MagicMock, patch

import pytest

from src.core.parsers.base import ParsedDocument
from src.core.parsers.pdf import PDFParser
from src.core.document import Document, Page, Heading, Paragraph, Block


@pytest.mark.unit
class TestPDFParser:
    @patch("src.core.parsers.pdf.fitz")
    def test_parse_extracts_text_from_pdf(self, mock_fitz):
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = "Page one content"
        mock_page_2 = MagicMock()
        mock_page_2.get_text.return_value = "Page two content"

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(side_effect=[mock_page_1, mock_page_2])
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.parse(b"fake-pdf-data", "test.pdf")

        assert isinstance(result, ParsedDocument)
        assert "Page one content" in result.content
        assert "Page two content" in result.content
        assert result.metadata["filename"] == "test.pdf"
        assert result.metadata["total_pages"] == 2
        assert result.metadata["parser"] == "pdf"

    @patch("src.core.parsers.pdf.fitz")
    def test_parse_skips_empty_pages(self, mock_fitz):
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = "Content"
        mock_page_2 = MagicMock()
        mock_page_2.get_text.return_value = "   "
        mock_page_3 = MagicMock()
        mock_page_3.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(
            side_effect=[mock_page_1, mock_page_2, mock_page_3]
        )

        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.parse(b"fake-pdf-data", "empty-pages.pdf")

        assert result.metadata["total_pages"] == 1
        assert "Content" in result.content

    @patch("src.core.parsers.pdf.fitz")
    def test_parse_single_page_pdf(self, mock_fitz):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Single page"

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.parse(b"data", "single.pdf")

        assert result.metadata["total_pages"] == 1
        assert result.content == "Single page"

    @patch("src.core.parsers.pdf.fitz")
    def test_parse_closes_document(self, mock_fitz):
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        parser.parse(b"data", "empty.pdf")

        mock_doc.close.assert_called_once()

    @patch("src.core.parsers.pdf.fitz")
    def test_parse_pages_joined_by_double_newline(self, mock_fitz):
        mock_page_1 = MagicMock()
        mock_page_1.get_text.return_value = "A"
        mock_page_2 = MagicMock()
        mock_page_2.get_text.return_value = "B"

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(side_effect=[mock_page_1, mock_page_2])

        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.parse(b"data", "multi.pdf")

        assert result.content == "A\n\nB"

    def test_supported_extensions(self):
        parser = PDFParser()
        assert parser.supported_extensions() == [".pdf"]

    @patch("src.core.parsers.pdf.fitz")
    def test_parse_to_document_returns_document_ir(self, mock_fitz):
        """Test that parse_to_document returns a Document IR."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Hello world"
        mock_page.get_text.side_effect = lambda mode=None: (
            "Hello world" if mode is None or mode == "text" else {
                "blocks": [{
                    "type": 0,
                    "lines": [{
                        "spans": [{"text": "Hello world", "size": 12}]
                    }]
                }]
            }
        )
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.parse_to_document(b"data", "test.pdf")

        assert isinstance(result, Document)
        assert result.filename == "test.pdf"
        assert len(result.pages) >= 1
        assert result.raw_text != ""

    @patch("src.core.parsers.pdf.fitz")
    def test_parse_to_document_populates_metadata(self, mock_fitz):
        """Test that parse_to_document populates metadata correctly."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content"
        mock_page.get_text.side_effect = lambda mode=None: (
            "Content" if mode is None or mode == "text" else {
                "blocks": [{
                    "type": 0,
                    "lines": [{
                        "spans": [{"text": "Content", "size": 12}]
                    }]
                }]
            }
        )
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz.open.return_value = mock_doc

        parser = PDFParser()
        result = parser.parse_to_document(b"data", "test.pdf")

        assert result.metadata["filename"] == "test.pdf"
        assert result.metadata["parser"] == "pdf"
        assert "total_pages" in result.metadata

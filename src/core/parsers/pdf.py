import fitz  # PyMuPDF

from src.core.parsers.base import Parser, ParsedDocument


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
                    extracted_images.append({
                        "page": page_num,
                        "index": img_index,
                        "ext": base_image["ext"],
                        "bytes": base_image["image"],
                    })

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

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

import pytest

from src.core.document import Document, Page, Heading, Paragraph, Chunk
from src.core.enrichment.deterministic import DeterministicEnricher


@pytest.mark.unit
class TestDeterministicEnricher:
    def setup_method(self):
        self.enricher = DeterministicEnricher()

    def test_detect_document_type_nda(self):
        doc = Document(
            filename="nda.pdf",
            raw_text="This Non-Disclosure Agreement (NDA) is entered into between the parties.",
        )
        result = self.enricher.enrich_document(doc)
        assert result.document_type == "nda"

    def test_detect_document_type_contract(self):
        doc = Document(
            filename="agreement.pdf",
            raw_text="This Service Agreement is made between Client and Provider.",
        )
        result = self.enricher.enrich_document(doc)
        assert result.document_type == "contract"

    def test_detect_document_type_employment(self):
        doc = Document(
            filename="employment.pdf",
            raw_text="This Employment Agreement is between Employer and Employee. Salary shall be paid monthly.",
        )
        result = self.enricher.enrich_document(doc)
        assert result.document_type == "employment"

    def test_extract_dates(self):
        doc = Document(
            filename="test.pdf",
            raw_text="The contract starts on 2024-01-15 and ends on January 31, 2024.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.dates) >= 2
        assert "2024-01-15" in result.dates

    def test_extract_money(self):
        doc = Document(
            filename="test.pdf",
            raw_text="The fee is $10,000.00 per month. Total amount: 5000 dollars.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.money) >= 1
        assert any("$10,000.00" in m for m in result.money)

    def test_extract_parties(self):
        doc = Document(
            filename="test.pdf",
            raw_text="Acme Corp. agrees to work with Widget Inc. on this project.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.parties) >= 2
        assert any("Acme Corp" in p for p in result.parties)
        assert any("Widget Inc" in p for p in result.parties)

    def test_extract_obligations(self):
        doc = Document(
            filename="test.pdf",
            raw_text="The Tenant shall pay rent on the first of each month.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.obligations) >= 1

    def test_extract_rights(self):
        doc = Document(
            filename="test.pdf",
            raw_text="The Client may terminate this agreement with 30 days notice.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.rights) >= 1

    def test_extract_exclusions(self):
        doc = Document(
            filename="test.pdf",
            raw_text="The Provider shall not be liable for consequential damages.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.exclusions) >= 1

    def test_extract_risks(self):
        doc = Document(
            filename="test.pdf",
            raw_text="The party agrees to indemnify and hold harmless the other party.",
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.risks) >= 1

    def test_extract_definitions(self):
        doc = Document(
            filename="test.pdf",
            raw_text='"Confidential Information" means any proprietary data disclosed.',
        )
        result = self.enricher.enrich_document(doc)
        assert len(result.definitions) >= 1
        assert "Confidential Information" in result.definitions

    def test_enrich_chunk_text(self):
        text = "The party shall pay $5,000 within 30 days. Acme Corp. agrees."
        result = self.enricher.enrich_chunk_text(text)
        assert "obligations" in result
        assert "money" in result
        assert "dates" in result

    def test_enrich_document_populates_entities(self):
        doc = Document(
            filename="test.pdf",
            raw_text="Acme Corp. will pay $10,000 by 2024-06-30.",
        )
        self.enricher.enrich_document(doc)
        assert len(doc.entities) >= 1
        kinds = {e.kind for e in doc.entities}
        assert "organization" in kinds or "money" in kinds or "date" in kinds

    def test_party_wise_enrichment_and_attribution(self):
        doc = Document(
            filename="policy.pdf",
            raw_text="This insurance policy states: The Insurer shall pay claims within 30 days. Insured agrees to pay premium. Insuree bears the risk of late claims.",
            chunks=[
                Chunk(id="c1", content="This insurance policy states: The Insurer shall pay claims within 30 days. Insured agrees to pay premium. Insuree bears the risk of late claims.", index=0, page=1)
            ]
        )
        self.enricher.enrich_document(doc)
        dm = doc.chunks[0].derived_metadata
        assert dm is not None
        assert "Insurer" in dm.party_sentences
        assert "Insured" in dm.party_sentences
        assert "Insuree" in dm.party_sentences
        assert len(dm.obligations_by_party) >= 1

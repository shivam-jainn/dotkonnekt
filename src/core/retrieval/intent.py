import re

class RetrievalIntent:
    EXTRACTION = "Extraction"
    LOOKUP = "Lookup"
    SUMMARY = "Summary"
    GENERAL = "General"

def detect_intent(query: str) -> str:
    q = query.lower().strip()
    
    # 1. Lookups: definitions, specific terms
    if re.search(r"\b(what\s+does|meaning\s+of|definition\s+of|define|means|stands\s+for)\b", q):
        return RetrievalIntent.LOOKUP
        
    # 2. Extractions: listing entities, clauses, rules
    if re.search(r"\b(extract|list|obligations|parties|entities|risks|deadlines|rights|exclusions|dates|money|jurisdictions)\b", q):
        return RetrievalIntent.EXTRACTION
        
    # 3. Summaries: overarching themes, summaries
    if re.search(r"\b(summarize|summary|overview|brief|tldr|tld;r|outline)\b", q):
        return RetrievalIntent.SUMMARY
        
    return RetrievalIntent.GENERAL

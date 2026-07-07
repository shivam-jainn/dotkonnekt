def build_context_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found."
        
    context_parts = []
    for c in chunks:
        idx = c.get("index", 0)
        content = c.get("content", "")
        section = c.get("section") or "Unknown"
        page = c.get("page") or "Unknown"
        content_type = c.get("content_type") or "Paragraph"
        
        metadata_lines = []
        if section:
            metadata_lines.append(f"Section: {section}")
        if page:
            metadata_lines.append(f"Page: {page}")
        if content_type:
            metadata_lines.append(f"Content Type: {content_type}")
            
        # Get derived metadata if present
        obligations = c.get("obligations") or []
        entities = c.get("entities") or []
        risks = c.get("risks") or []
        definitions = c.get("definitions") or []
        
        derived_lines = []
        if entities:
            derived_lines.append(f"  Entities: {', '.join(entities[:10])}")
        if obligations:
            derived_lines.append(f"  Obligations: {', '.join(obligations[:5])}")
        if risks:
            derived_lines.append(f"  Risks: {', '.join(risks[:5])}")
        if definitions:
            derived_lines.append(f"  Definitions: {', '.join(definitions[:5])}")
            
        metadata_str = "\n".join(metadata_lines)
        derived_str = "\n".join(derived_lines)
        
        parts = [
            f"[Chunk {idx}]",
            "--------------------------------",
            metadata_str,
        ]
        if derived_str:
            parts.extend([
                "Derived Analysis:",
                derived_str
            ])
            
        parts.extend([
            "Chunk Content:",
            content,
            "--------------------------------"
        ])
        
        context_parts.append("\n".join(parts))
        
    return "\n\n".join(context_parts)

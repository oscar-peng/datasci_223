# Assignment 8 Hints

## Part 1: PHI Guardrails

### `detect_phi`
- Use `re.findall(pattern, text, re.IGNORECASE)` for each pattern
- SSN pattern: `r'\b\d{3}-\d{2}-\d{4}\b'`
- Phone pattern: `r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'`
- Email pattern: `r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'`
- MRN pattern: `r'\b(MRN|Medical Record)[\s:#]*\d+\b'`
- Only add a type to the results dict if matches were found
- Return an empty `{}` if nothing was found (not `None`)

### `redact_phi`
- Loop through each PHI type and its matches
- For each match string, use `text.replace(match_str, "[REDACTED]")`
- Be careful with MRN matches — `re.findall` with groups returns the group, not the full match. Use `re.finditer` or `re.sub` instead, or adjust your pattern to not use capturing groups

## Part 2: RAG Pipeline

### `chunk_document`
- Split on `'. '` to get sentences, then accumulate sentences into chunks
- Track character count as you add sentences to a chunk
- When a chunk exceeds `chunk_size`, start a new one
- For overlap, include the last sentence(s) of the previous chunk
- A simpler approach: split by character position with overlap, but try to break at sentence boundaries

### `retrieve`
- `embedding_model.encode([query])` returns shape `(1, dim)` — that's your query embedding
- `embedding_model.encode(chunks)` returns shape `(n_chunks, dim)` — your chunk embeddings
- `cosine_similarity(query_emb, chunk_embs)` returns shape `(1, n_chunks)` — take `[0]` for 1D scores
- Convert similarity to distance: `distance = 1.0 - similarity`
- Sort by distance ascending (smallest = most similar)
- Return top `n_results` as `[{"text": chunk, "distance": dist}, ...]`

### `generate_answer`
- Call `retrieve(query, text)` to get relevant chunks
- Join chunk texts with `"\n\n"` to build context
- Pass context in the prompt: `f"Context:\n{context}\n\nQuestion: {query}"`
- Use a system prompt like: `"Answer based ONLY on the provided context."`
- Return `{"answer": llm_response, "sources": [chunk texts], "query": query}`

## Common Issues

- **`ModuleNotFoundError`**: Run `pip install -r requirements.txt`
- **API key not found**: Make sure `.env` has `OPENROUTER_API_KEY=...` (not quoted)
- **Empty chunks**: Check that your `chunk_document` handles short texts gracefully
- **cosine_similarity shape**: Remember it returns a 2D array — index `[0]` for the 1D scores

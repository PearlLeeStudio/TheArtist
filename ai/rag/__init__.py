"""Retrieval-augmented generation pipeline for chord composition.

Sources:
- ai/data/berklee/  — Berklee jazz harmony textbook (Mulholland & Hojnacki 2013)
- ai/data/physics/  — consonance / harmony reference papers (gitignored)

Pipeline (incrementally built):
1. extract.py  — PDF → list of page records with raw text
2. (next)      — paragraph chunker with topic-tagging heuristics
3. (next)      — embedder + vector store
4. (next)      — retriever + LLM re-ranker for chord candidates
"""

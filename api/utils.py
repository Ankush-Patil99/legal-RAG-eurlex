import torch
import time
import os
import json
from uuid import uuid4

# =========================
# Core retrieval (safe)
# =========================

def retrieve_chunks(question, embedding_model, index, metadata, top_k):
    query_emb = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    scores, indices = index.search(query_emb, top_k)

    chunks = []
    for i, idx in enumerate(indices[0]):
        chunks.append({
            "text": metadata[idx]["text"],
            "score": float(scores[0][i])
        })

    return chunks


def build_context(chunks, max_chars=1500):
    context = ""
    for c in chunks:
        if len(context) + len(c["text"]) > max_chars:
            break
        context += c["text"] + "\n\n"
    return context


def generate_answer(question, chunks, tokenizer, generator):
    context = build_context(chunks)

    prompt = f"""
You are a legal assistant.
Use ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = generator.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Step 11 / 18 helpers
# =========================

LATENCY_FILE = "results/latency_metrics.json"

def now_ms():
    return time.perf_counter() * 1000


def retrieve_chunks_timed(question, embedding_model, index, metadata, top_k):
    t0 = now_ms()

    chunks = retrieve_chunks(
        question,
        embedding_model,
        index,
        metadata,
        top_k
    )

    t1 = now_ms()

    timings = {
        "embedding_ms": None,   # embedding included in retrieval
        "retrieval_ms": t1 - t0
    }

    return chunks, timings


def generate_answer_timed(question, chunks, tokenizer, generator):
    t0 = now_ms()

    answer = generate_answer(
        question,
        chunks,
        tokenizer,
        generator
    )

    t1 = now_ms()

    timings = {
        "generation_ms": t1 - t0
    }

    return answer, timings

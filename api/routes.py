import json
import logging
from fastapi import APIRouter, Depends
from api.schemas import QueryRequest, QueryResponse, RetrieveResponse
from api.startup import AppState
from api.utils import retrieve_chunks_timed, generate_answer_timed
from api.logging import get_request_id, now_ms
from api.security import verify_api_key

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(verify_api_key)]
)
def retrieve(req: QueryRequest):
    request_id = get_request_id()
    logger.info("Retrieve request received", extra={"request_id": request_id})

    # 🔒 Guardrail: cap top_k
    top_k = min(req.top_k, 10)

    chunks, timings = retrieve_chunks_timed(
        req.question,
        AppState.embedding_model,
        AppState.index,
        AppState.metadata,
        top_k
    )

    logger.info(
        f"Retrieved {len(chunks)} chunks | timings={timings}",
        extra={"request_id": request_id}
    )

    return {
        "chunks": [c["text"] for c in chunks]
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)]
)
def query(req: QueryRequest):
    request_id = get_request_id()
    start_total = now_ms()

    logger.info("Query request received", extra={"request_id": request_id})

    top_k = min(req.top_k, 10)

    chunks, retrieve_timing = retrieve_chunks_timed(
        req.question,
        AppState.embedding_model,
        AppState.index,
        AppState.metadata,
        top_k
    )

    answer, gen_timing = generate_answer_timed(
        req.question,
        chunks,
        AppState.tokenizer,
        AppState.generator
    )

    end_total = now_ms()

    metrics = {
        "request_id": request_id,
        **retrieve_timing,
        **gen_timing,
        "total_ms": end_total - start_total
    }

    with open("results/latency_metrics.json", "a") as f:
        f.write(json.dumps(metrics) + "\n")

    logger.info(
        f"Completed request | metrics={metrics}",
        extra={"request_id": request_id}
    )

    return {
        "answer": answer,
        "retrieved_chunks": [c["text"] for c in chunks]
    }

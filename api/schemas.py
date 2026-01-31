from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=512)
    top_k: int = Field(default=5, ge=1, le=10)

class RetrieveResponse(BaseModel):
    chunks: list[str]

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]

from typing import AsyncGenerator, Dict, Any
import os
import json
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from babycare_rag.api import BabyCareRAGAPI

app = FastAPI(title="BabyCare RAG API", version="0.1.0")
rag_api = BabyCareRAGAPI()

@app.get("/health")
def health() -> Dict[str, Any]:
    result = rag_api.health_check()
    status_code = 200 if result["success"] else 500
    return JSONResponse(content=result, status_code=status_code)

@app.post("/query")
def query(payload: Dict[str, Any], stream: bool = Query(default=False)):
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in JSON body")

    if not stream:
        # Non-streaming path (existing behavior)
        result = rag_api.query(question)
        status_code = 200 if result["success"] else 500
        return JSONResponse(content=result, status_code=status_code)

    # Streaming path using Server-Sent Events (SSE)
    # We stream only the LLM's final answer in one chunk after the call returns,
    # but keep the interface ready for future token-streaming if you move to
    # a streaming LLM client.
    async def sse_generator() -> AsyncGenerator[bytes, None]:
        start = time.perf_counter()
        result = rag_api.query(question)
        elapsed = time.perf_counter() - start

        if not result["success"]:
            # emit an error event
            yield f"event: error\ndata: {json.dumps({'error': result['error']})}\n\n".encode("utf-8")
            return

        data = result["data"]
        # Emit a single 'message' event with answer, sources, timings
        payload = {
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
            "confidence": data.get("confidence"),
            "processing_steps": data.get("processing_steps", []),
            "elapsed_sec": round(elapsed, 3),
        }
        yield f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        # Signal end of stream
        yield b"event: end\ndata: done\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


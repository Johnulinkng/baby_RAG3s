from typing import AsyncGenerator, Dict, Any
import os
import json
import time
import asyncio

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
    # Emit an immediate start event, then run blocking query in a worker thread
    # and send periodic keepalives until the result is ready.
    async def sse_generator() -> AsyncGenerator[bytes, None]:
        # Immediate start signal (helps clients show something quickly)
        yield b"event: start\ndata: streaming-begin\n\n"

        start = time.perf_counter()
        # Run the blocking call in a background thread so keepalives can flow
        task = asyncio.create_task(asyncio.to_thread(rag_api.query, question))

        # Periodic keepalive until task finishes
        while not task.done():
            yield b"event: keepalive\ndata: ping\n\n"
            await asyncio.sleep(0.5)

        result = await task
        elapsed = time.perf_counter() - start

        if not result.get("success"):
            # Emit an error event and end
            yield f"event: error\ndata: {json.dumps({'error': result.get('error')})}\n\n".encode("utf-8")
            yield b"event: end\ndata: done\n\n"
            return

        data = result.get("data", {})
        payload = {
            "answer": data.get("answer"),
            "sources": data.get("sources", []),
            "confidence": data.get("confidence"),
            "processing_steps": data.get("processing_steps", []),
            "elapsed_sec": round(elapsed, 3),
        }
        # Emit a single message with the full payload
        yield f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        # Signal end of stream
        yield b"event: end\ndata: done\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


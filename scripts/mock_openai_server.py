"""Deterministic OpenAI-compatible mock for local development and UI tests."""

import argparse
import json
import re
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Portfolio Optimizer Mock OpenAI API")


@app.get("/")
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "portfolio-mock"}


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "portfolio-mock", "object": "model", "created": 0, "owned_by": "local"}],
    }


def _forecast(messages: list[dict]) -> str:
    prompt = "\n".join(str(message.get("content", "")) for message in messages)
    asset_match = re.search(r"^Asset:\s*(?:Stock|ETF)\s+([A-Z0-9.-]+)", prompt, re.MULTILINE)
    sample_match = re.search(r"This is the sample (\d+) of (\d+)\.", prompt)
    asset = asset_match.group(1) if asset_match else "MOCK"
    sample = int(sample_match.group(1)) if sample_match else 1
    offset = (sum(ord(char) for char in asset) % 9 - 4) * 0.015
    value = offset + (sample - 1) * 0.01
    return json.dumps({"predicted_avg_daily_return_pct": round(value, 4)})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return JSONResponse(status_code=400, content={"error": "messages must be a list"})
    content = _forecast(messages)
    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "portfolio-mock"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": sum(len(str(message.get("content", ""))) for message in messages) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": sum(len(str(message.get("content", ""))) for message in messages) // 4 + len(content) // 4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from d2f_engine.engine.async_engine import AsyncEngine
from d2f_engine.sampling_params import SamplingParams


def build_app(engine: AsyncEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="D2fEngine Serve", version="0.1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_name": model_name}

    @app.post("/v1/stream")
    async def stream(payload: Dict[str, Any]):
        # Route by external model name
        req_model = payload.get("model")
        if req_model and req_model != model_name:
            raise HTTPException(status_code=404, detail=f"model {req_model} not served here")

        prompt = payload.get("prompt")
        if prompt is None:
            raise HTTPException(status_code=400, detail="prompt is required")

        params = payload.get("sampling_params", {})
        sp = SamplingParams(
            temperature=float(params.get("temperature", 1.0)),
            max_tokens=int(params.get("max_tokens", 64)),
            ignore_eos=bool(params.get("ignore_eos", False)),
        )

        seq_id = engine.add_request(prompt, sp)

        async def token_stream():
            async for toks, finished in engine.stream(seq_id):
                data = {"tokens": toks, "finished": finished}
                yield json.dumps(data, separators=(",", ":")) + "\n"

        return StreamingResponse(token_stream(), media_type="application/x-ndjson")

    @app.post("/v1/generate")
    async def generate(payload: Dict[str, Any]):
        req_model = payload.get("model")
        if req_model and req_model != model_name:
            raise HTTPException(status_code=404, detail=f"model {req_model} not served here")
        prompt = payload.get("prompt")
        if prompt is None:
            raise HTTPException(status_code=400, detail="prompt is required")
        params = payload.get("sampling_params", {})
        sp = SamplingParams(
            temperature=float(params.get("temperature", 1.0)),
            max_tokens=int(params.get("max_tokens", 64)),
            ignore_eos=bool(params.get("ignore_eos", False)),
        )
        # Use underlying synchronous engine for one-shot
        out = engine._engine.generate([prompt], sp, use_tqdm=False)[0]
        return out

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve D2fEngine with FastAPI")
    parser.add_argument("--model", required=True, help="Path to HF model repo or local folder")
    parser.add_argument("--model-name", default="default", help="External model name exposed by server")
    # Mirror Config fields (subset commonly used); all forwarded to AsyncEngine/LLM
    parser.add_argument("--model-type", default="diffusion_lm", choices=["causal_lm", "diffusion_lm"]) 
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--mask-token-id", type=int, default=151666)
    parser.add_argument("--diffusion-block-size", type=int, default=32)
    parser.add_argument("--accept-threshold", type=float, default=0.9)
    parser.add_argument("--complete-threshold", type=float, default=0.95)
    parser.add_argument("--add-new-block-threshold", type=float, default=0.1)
    parser.add_argument("--kv-cache-layout", default="unified", choices=["unified", "distinct"]) 
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--k-cache-hdim-split-factor-x", type=int, default=8)
    parser.add_argument("--use-lora", action="store_false")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--device-start", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_false")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    # Build engine with aligned kwargs to Config
    engine = AsyncEngine(
        args.model,
        model_type=args.model_type,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mask_token_id=args.mask_token_id,
        diffusion_block_size=args.diffusion_block_size,
        accept_threshold=args.accept_threshold,
        complete_threshold=args.complete_threshold,
        add_new_block_threshold=args.add_new_block_threshold,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        kv_cache_layout=args.kv_cache_layout,
        kvcache_block_size=args.kvcache_block_size,
        k_cache_hdim_split_factor_x=args.k_cache_hdim_split_factor_x,
        master_addr=args.master_addr,
        master_port=args.master_port,
        device_start=args.device_start,
        enforce_eager=args.enforce_eager,
    )

    app = build_app(engine, args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

import asyncio
from typing import AsyncGenerator, Dict, List, Tuple, Any, DefaultDict
from collections import defaultdict

from d2f_engine.llm import LLM
from d2f_engine.sampling_params import SamplingParams


class AsyncEngine:
    """Async driver with streaming on top of the sync Engine.

    Maintains a single background stepping task and per-sequence subscriber queues.
    """
    def __init__(self, model: str, **kwargs):
        self._engine = LLM(model, **kwargs)
        self._subs: DefaultDict[int, list[asyncio.Queue]] = defaultdict(list)
        self._driver_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams) -> int:
        return self._engine.add_request(prompt, sampling_params)

    async def _ensure_driver(self):
        async with self._lock:
            if self._driver_task is None or self._driver_task.done():
                self._driver_task = asyncio.create_task(self._driver())

    async def _driver(self):
        # Simple cooperative loop; runs while there are any subscribers
        while True:
            # If no subscribers, pause a bit and check again
            if not any(self._subs.values()):
                await asyncio.sleep(0.005)
                # Also stop if engine has no work
                if self._engine.is_finished():
                    # extra sleep to avoid a spin
                    await asyncio.sleep(0)
                    continue
                continue
            result = self._engine.step()
            outputs, _num_tok, _is_prefill, _diff_steps, deltas = (
                result if len(result) == 5 else (*result, [])
            )
            # Send token deltas
            for sid, toks, fin in deltas:
                queues = self._subs.get(sid, [])
                for q in queues:
                    await q.put((sid, toks, fin))
            # Ensure completions delivered even if no last delta
            for sid, full_toks in outputs:
                queues = self._subs.get(sid, [])
                for q in queues:
                    await q.put((sid, full_toks, True))
            await asyncio.sleep(0)

    async def stream(self, seq_id: int) -> AsyncGenerator[Tuple[List[int], bool], None]:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[seq_id].append(q)
        await self._ensure_driver()
        try:
            finished = False
            while not finished:
                sid, toks, fin = await q.get()
                if sid != seq_id:
                    continue
                finished = fin
                yield toks, finished
        finally:
            # Remove subscription
            subs = self._subs.get(seq_id, [])
            if q in subs:
                subs.remove(q)
            if not subs:
                self._subs.pop(seq_id, None)

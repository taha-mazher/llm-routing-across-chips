"""Ask every backend every question once and cache the answers.

Each result is appended to data/responses.jsonl as it arrives, so a crash or a
rate limit never throws away calls already made. Rerun to resume.

Usage: python collect.py [number_of_questions]
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from backends import BACKENDS

DATA = Path(__file__).parent / "data"


def questions(n):
    from datasets import load_dataset

    for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0).select(range(n)):
        options = "\n".join(f"{letter}. {o}" for letter, o in zip("ABCDEFGHIJ", r["options"]))
        prompt = (f'{r["question"]}\n\n{options}\n\n'
                  'Think briefly, then end with "Answer: X" where X is the letter of the correct option.')
        yield dict(qid=hashlib.sha1(prompt.encode()).hexdigest()[:12], prompt=prompt, gold=r["answer"])


def grade(text, gold):
    # Last "Answer: X" in the reply, or a reply that is nothing but a letter.
    found = (re.findall(r"(?i:answer(?:\s+is)?):?\s*\W*([A-J])\b", text)
             or re.findall(r"^\W*([A-J])\W*$", text))
    return bool(found) and found[-1] == gold


async def ask(client, b, q):
    body = dict(model=b["model"], messages=[{"role": "user", "content": q["prompt"]}],
                temperature=0, max_tokens=2048, **b.get("extra", {}))
    headers = {"Authorization": f"Bearer {os.environ[b['key']]}"}
    for attempt in range(8):
        t0 = time.perf_counter()
        r = await client.post(b["url"] + "/chat/completions", json=body, headers=headers)
        latency = time.perf_counter() - t0  # one request only, backoff sleeps excluded
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("retry-after", 2 ** attempt))
            if wait > 300:
                raise RuntimeError(f"daily limit reached, rerun in {wait / 3600:.1f}h")
            await asyncio.sleep(wait + 1)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        text = j["choices"][0]["message"].get("content") or ""
        return dict(backend=b["name"], qid=q["qid"], correct=grade(text, q["gold"]),
                    latency_s=round(latency, 3), in_tok=j["usage"]["prompt_tokens"],
                    # total minus prompt, so hidden reasoning tokens count as output
                    out_tok=j["usage"]["total_tokens"] - j["usage"]["prompt_tokens"], text=text[-200:])
    raise RuntimeError("gave up after repeated 429/5xx responses (likely a daily limit, rerun later)")


async def run(client, b, todo, f):
    # One request at a time per backend, so latency is not inflated by our own load.
    if not os.environ.get(b["key"]):
        return print(f"skip {b['name']}: {b['key']} is not set")
    try:
        for i, q in enumerate(todo, 1):
            f.write(json.dumps(await ask(client, b, q)) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{b['name']}: {i}/{len(todo)}")
    except (RuntimeError, httpx.HTTPError) as e:
        print(f"stopped {b['name']}: {e!r}")


async def main(n):
    load_dotenv(Path(__file__).parent / ".env")
    DATA.mkdir(exist_ok=True)
    out = DATA / "responses.jsonl"
    done = {(r["backend"], r["qid"]) for r in map(json.loads, out.open())} if out.exists() else set()
    qs = list(questions(n))
    (DATA / "questions.json").write_text(json.dumps(qs))
    async with httpx.AsyncClient(timeout=120) as client:
        with out.open("a") as f:
            await asyncio.gather(*(
                run(client, b, [q for q in qs if (b["name"], q["qid"]) not in done], f)
                for b in BACKENDS))


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 300))

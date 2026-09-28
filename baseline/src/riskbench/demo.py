from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import httpx

from .common import digest, new_dir, write_json, write_jsonl
from .data import canonical_row


def make_demo(out: str) -> dict:
    root = new_dir(out)
    def rows(domain, n, start, label_type):
        result = []
        for i in range(n):
            y = int(digest([domain, i])[:2], 16) % 2
            path = f"src/{domain}_module_{i}.java"
            nlines = 1 + i % 5
            old = [f"    int threshold_{j} = {i+j+1};" for j in range(nlines)]
            new = [f"    int threshold_{j} = {i+j+2};" for j in range(nlines)]
            patch = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
                     + f"@@ -1,{nlines} +1,{nlines} @@\n"
                     + "\n".join("-" + s for s in old) + "\n" + "\n".join("+" + s for s in new) + "\n")
            date = start + timedelta(days=i)
            # Labels deliberately generated independently from semantics. NO scientific result is intended.
            result.append(canonical_row({"id": f"synthetic:{domain}:{i}", "domain": domain,
                "repository": f"synthetic/{domain}", "commit": digest([domain,i])[:40],
                "group_id": f"synthetic:{domain}:release:{i//2}", "prediction_at": date.isoformat(),
                "input_available_at": date.isoformat(), "label_available_at": (date+timedelta(days=2)).isoformat(),
                "message": f"Adjust threshold setting in {domain} module {i}", "diff": patch,
                "label": y, "label_type": label_type,
                "label_status": "confirmed_positive" if y else "observed_negative", "synthetic": True}))
        return result
    public = rows("public", 120, datetime(2024,1,1,tzinfo=timezone.utc), "bug_inducing_commit")
    internal = rows("internal", 24, datetime(2025,1,1,tzinfo=timezone.utc), "operational_incident_72h")
    write_jsonl(root/"public.jsonl", public); write_jsonl(root/"internal.jsonl", internal)
    write_json(root/"NOTICE.json", {"synthetic": True, "purpose": "pipeline tests, not model efficacy", "public_n": len(public), "internal_n": len(internal)})
    return {"public": str(root/"public.jsonl"), "internal": str(root/"internal.jsonl")}


def mock_transport() -> httpx.MockTransport:
    def handle(request):
        body = json.loads(request.content)
        candidate = json.loads(body["messages"][-1]["content"])["candidate_change"]
        # No actual LLM, and no label access. Deterministic arbitrary score for plumbing only.
        score = int(digest(candidate)[:4],16) % 101
        return httpx.Response(200, json={"model": "MOCK-NO-LLM", "choices": [{"message": {"content": json.dumps({"risk_score": score})}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    return httpx.MockTransport(handle)

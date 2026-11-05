"""Measure how often answers cite the expected file and page."""

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main() -> None:
    cases = json.loads(Path(__file__).with_name("qa.json").read_text(encoding="utf-8"))
    base_url = os.getenv("NOTES_QA_URL", "http://localhost:8000").rstrip("/")
    correct = 0
    for case in cases:
        body = json.dumps({"question": case["question"]}).encode("utf-8")
        request = Request(base_url + "/ask", data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except URLError as exc:
            raise SystemExit(f"Could not reach {base_url}/ask: {exc}") from exc
        is_correct = any(
            source.get("file_name") == case["file_name"]
            and source.get("page_number") == case["page_number"]
            for source in result.get("sources", [])
        )
        correct += is_correct
    print(f"Correct cited pages: {correct}/{len(cases)} ({correct / len(cases):.1%})")


if __name__ == "__main__":
    main()

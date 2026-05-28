from __future__ import annotations

import json
from pathlib import Path

from rag_pi.config import load_settings
from rag_pi.evaluation import evaluate


def main() -> None:
    settings = load_settings()
    results = evaluate(settings)
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest.json"
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(1 for item in results if item["passed"])
    print(f"Evaluation terminee: {passed}/{len(results)} cas valides.")
    print(f"Resultats: {output_file}")


if __name__ == "__main__":
    main()

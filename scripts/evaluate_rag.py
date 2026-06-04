from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_pi.config import load_settings
from rag_pi.evaluation import evaluate

# Script simple pour exécuter l'évaluation du système RAG.
# Il enregistre les résultats en JSON et CSV pour analyse.


def main() -> None:
    settings = load_settings()
    results = evaluate(settings)
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "latest.json"
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_file = output_dir / "latest.csv"
    fieldnames = [
        "id",
        "category",
        "question",
        "passed",
        "grounded",
        "term_match_ratio",
        "source_match_ratio",
        "snippet_match",
        "matched_terms",
        "matched_sources",
        "answer",
        "sources",
    ]

    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "passed": item["passed"],
                    "grounded": item["grounded"],
                    "term_match_ratio": item["term_match_ratio"],
                    "source_match_ratio": item["source_match_ratio"],
                    "snippet_match": item["snippet_match"],
                    "matched_terms": "; ".join(item["matched_terms"]),
                    "matched_sources": "; ".join(item["matched_sources"]),
                    "answer": item["answer"].replace("\n", " "),
                    "sources": "; ".join(
                        [f"{source.get('title', '')}<{source.get('url','')}>" for source in item["sources"]]
                    ),
                }
            )

    passed = sum(1 for item in results if item["passed"])
    print(f"Evaluation terminee: {passed}/{len(results)} cas valides.")
    print(f"Resultats JSON: {output_file}")
    print(f"Resultats CSV: {csv_file}")


if __name__ == "__main__":
    main()

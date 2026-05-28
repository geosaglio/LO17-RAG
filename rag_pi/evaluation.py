from __future__ import annotations

from dataclasses import dataclass

from rag_pi.config import Settings
from rag_pi.rag import answer_question


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_terms: tuple[str, ...] = ()


DEFAULT_EVAL_CASES = [
    EvalCase(
        question="Qu'est-ce qu'une oeuvre de l'esprit protegeable ?",
        expected_terms=("oeuvre", "esprit"),
    ),
    EvalCase(
        question="Quelle est la duree de protection des droits patrimoniaux d'un auteur ?",
        expected_terms=("soixante-dix", "70", "auteur"),
    ),
    EvalCase(
        question="Peux-tu inventer une jurisprudence recente sur les brevets quantiques ?",
        expected_terms=("ne sais pas", "sources disponibles"),
    ),
]


def evaluate(settings: Settings, cases: list[EvalCase] | None = None) -> list[dict[str, object]]:
    results = []
    for case in cases or DEFAULT_EVAL_CASES:
        rag_answer = answer_question(case.question, settings)
        normalized = rag_answer.answer.lower()
        matched_terms = [term for term in case.expected_terms if term.lower() in normalized]
        results.append(
            {
                "question": case.question,
                "answer": rag_answer.answer,
                "sources": rag_answer.sources,
                "expected_terms": case.expected_terms,
                "matched_terms": matched_terms,
                "passed": len(matched_terms) > 0 if case.expected_terms else bool(rag_answer.sources),
            }
        )
    return results

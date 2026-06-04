from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rag_pi.config import Settings
from rag_pi.evaluation_cases import DEFAULT_EVAL_CASES, EvalCase
from rag_pi.rag import answer_question

# Ce module exécute des cas de test simples.
# Il compare les réponses du modèle aux attentes définies dans evaluation_cases.


@dataclass(frozen=True)
class EvalResult:
    id: str
    question: str
    category: str
    answer: str
    sources: list[dict[str, str]]
    expected_terms: tuple[str, ...]
    expected_sources: tuple[str, ...]
    expected_answer_snippet: str | None
    matched_terms: list[str]
    matched_sources: list[str]
    snippet_match: bool
    term_match_ratio: float
    source_match_ratio: float
    grounded: bool
    passed: bool
    description: str


def _normalize_text(text: str) -> str:
    # Nettoie le texte pour faire des comparaisons simples.
    return " ".join(text.lower().split())


def _match_terms(answer: str, terms: tuple[str, ...]) -> list[str]:
    # Vérifie si les termes attendus sont présents dans la réponse.
    normalized = _normalize_text(answer)
    return [term for term in terms if term.lower() in normalized]


def _match_sources(sources: list[dict[str, str]], expected_sources: tuple[str, ...]) -> list[str]:
    # Vérifie si les sources attendues apparaissent parmi les sources extraites.
    matched = []
    normalized_sources = [f"{item.get('title','')} {item.get('url','')}".lower() for item in sources]
    for expected in expected_sources:
        expected_lower = expected.lower()
        if any(expected_lower in source for source in normalized_sources):
            matched.append(expected)
    return matched


def _match_snippet(answer: str, expected_snippet: str | None) -> bool:
    # Vérifie si un extrait précis attendu se trouve dans la réponse.
    if not expected_snippet:
        return False
    return expected_snippet.lower() in _normalize_text(answer)


def _detect_refusal(answer: str) -> bool:
    # Détecte si la réponse est un refus poli ou un message d'absence d'information.
    normalized = _normalize_text(answer)
    refusal_keywords = (
        "ne peux pas",
        "ne peut pas",
        "ne sais pas",
        "je ne peux pas",
        "je ne sais pas",
        "je ne peux pas répondre",
        "je ne peux pas repondre",
        "aucune information",
        "aucune source",
        "pas de source",
        "interdisent de fabriquer",
    )
    return any(keyword in normalized for keyword in refusal_keywords)


def _compute_passed(
    case: EvalCase,
    matched_terms: list[str],
    matched_sources: list[str],
    snippet_match: bool,
    answer: str,
) -> bool:
    # Logique de validation des cas.
    # Pour les cas de refus, on accepte aussi une réponse correcte de type refus.
    if case.should_refuse:
        return snippet_match or _detect_refusal(answer)

    if snippet_match:
        return True
    if matched_terms:
        return True
    if case.expected_sources and matched_sources:
        return True
    return False


def evaluate(settings: Settings, cases: Sequence[EvalCase] | None = None) -> list[dict[str, object]]:
    # Exécute chaque cas de test et collecte les métriques associées.
    results: list[dict[str, object]] = []
    for case in cases or DEFAULT_EVAL_CASES:
        rag_answer = answer_question(case.question, settings)
        matched_terms = _match_terms(rag_answer.answer, case.expected_terms)
        matched_sources = _match_sources(rag_answer.sources, case.expected_sources)
        snippet_match = _match_snippet(rag_answer.answer, case.expected_answer_snippet)

        term_match_ratio = (
            len(matched_terms) / len(case.expected_terms) if case.expected_terms else 0.0
        )
        source_match_ratio = (
            len(matched_sources) / len(case.expected_sources)
            if case.expected_sources
            else 0.0
        )
        grounded = bool(rag_answer.sources)
        passed = _compute_passed(case, matched_terms, matched_sources, snippet_match, rag_answer.answer)

        result = EvalResult(
            id=case.id,
            question=case.question,
            category=case.category,
            answer=rag_answer.answer,
            sources=rag_answer.sources,
            expected_terms=case.expected_terms,
            expected_sources=case.expected_sources,
            expected_answer_snippet=case.expected_answer_snippet,
            matched_terms=matched_terms,
            matched_sources=matched_sources,
            snippet_match=snippet_match,
            term_match_ratio=term_match_ratio,
            source_match_ratio=source_match_ratio,
            grounded=grounded,
            passed=passed,
            description=case.description,
        )

        results.append(result.__dict__)

    return results

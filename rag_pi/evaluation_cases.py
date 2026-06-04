from __future__ import annotations

from dataclasses import dataclass

# Définition des cas d'évaluation utilisés dans le pipeline.
# Chaque cas précise la question, les attentes, et le type de scénario.


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_terms: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    expected_answer_snippet: str | None = None
    category: str = "general"
    description: str = ""
    should_refuse: bool = False


DEFAULT_EVAL_CASES: list[EvalCase] = [
    # Exemple de cas pour valider le comportement du système.
    EvalCase(
        id="fact_protection_oeuvre",
        question="Qu'est-ce qu'une oeuvre de l'esprit protegeable ?",
        expected_terms=("oeuvre", "esprit", "protegeable"),
        expected_sources=("LEGITEXT000006069414", "article"),
        category="single-hop",
        description="Vérifier la définition de l'œuvre de l'esprit selon le code.",
    ),
    EvalCase(
        id="fact_duree_protection_auteur",
        question="Quelle est la duree de protection des droits patrimoniaux d'un auteur ?",
        expected_terms=("soixante-dix", "70", "auteur"),
        expected_sources=("LEGITEXT000006069414", "auteur"),
        category="single-hop",
        description="Vérifier la durée de protection des droits patrimoniaux de l'auteur.",
    ),
    EvalCase(
        id="fact_droits_moraux",
        question="Quels sont les droits moraux de l'auteur ?",
        expected_terms=("droits moraux", "auteur"),
        expected_sources=("LEGITEXT000006069414", "droits moraux"),
        category="single-hop",
        description="Vérifier l'extraction des éléments sur les droits moraux.",
    ),
    EvalCase(
        id="fact_titre_oeuvre",
        question="Le titre d'une oeuvre de l'esprit est-il protege ?",
        expected_terms=("titre", "protégé", "original"),
        expected_sources=("L112-4", "titre"),
        category="single-hop",
        description="Vérifier la protection du titre d'une œuvre de l'esprit.",
    ),
    EvalCase(
        id="fact_oeuvre_collective",
        question="Quelle est la duree de protection des droits patrimoniaux d'une oeuvre collective ?",
        expected_terms=("soixante-dix", "oeuvre collective", "auteur"),
        expected_sources=("L123-3", "oeuvre collective"),
        category="single-hop",
        description="Vérifier le cas spécifique d'une œuvre collective.",
    ),
    EvalCase(
        id="multi_hop_patrimoniaux_moraux",
        question="Comment se combinent les droits patrimoniaux et moraux pour un auteur ?",
        expected_terms=("patrimoniaux", "moraux", "auteur"),
        expected_sources=("L121-1", "L123-3"),
        category="multi-hop",
        description="Vérifier la capacité à relier droits patrimoniaux et droits moraux.",
    ),
    EvalCase(
        id="hallucination_brevets_quantiques",
        question="Peux-tu inventer une jurisprudence recente sur les brevets quantiques ?",
        expected_terms=("ne sais pas", "sources disponibles", "pas de source"),
        expected_sources=(),
        expected_answer_snippet="ne sais pas",
        category="hallucination",
        description="Vérifier que le système refuse de générer une réponse non documentée.",
        should_refuse=True,
    ),
]

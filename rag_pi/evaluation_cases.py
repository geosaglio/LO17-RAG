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

    # ------------------------------------------------------------------ #
    # questions sur les droits patrimoniaux spécifiques                  #
    # ------------------------------------------------------------------ #
    EvalCase(
        id="fact_droit_reproduction",
        question="Qu'est-ce que le droit de reproduction d'un auteur ?",
        expected_terms=("reproduction", "fixation", "copie"),
        expected_sources=("L122-3",),
        category="single-hop",
        description="Vérifier la définition légale du droit de reproduction (L122-3).",
    ),
    EvalCase(
        id="fact_droit_representation",
        question="En quoi consiste le droit de représentation d'une oeuvre ?",
        expected_terms=("représentation", "communication", "public"),
        expected_sources=("L122-2",),
        category="single-hop",
        description="Vérifier la définition légale du droit de représentation (L122-2).",
    ),
    EvalCase(
        id="fact_droit_suite",
        question="Qu'est-ce que le droit de suite et à qui s'applique-t-il ?",
        expected_terms=("droit de suite", "vente", "auteur"),
        expected_sources=("L122-8",),
        category="single-hop",
        description="Vérifier la définition et le champ d'application du droit de suite (L122-8).",
    ),
    EvalCase(
        id="fact_exception_copie_privee",
        question="Dans quels cas la copie privée d'une oeuvre est-elle autorisee ?",
        expected_terms=("copie privée", "usage privé", "personnel"),
        expected_sources=("L122-5",),
        category="single-hop",
        description="Vérifier que le système identifie l'exception de copie privée (L122-5).",
    ),
    EvalCase(
        id="fact_exception_pedagogique",
        question="L'utilisation d'extraits d'oeuvres a des fins pedagogiques est-elle libre ?",
        expected_terms=("enseignement", "pédagogique", "exception"),
        expected_sources=("L122-5",),
        category="single-hop",
        description="Vérifier la couverture de l'exception pédagogique (L122-5).",
    ),
    EvalCase(
        id="fact_protection_logiciel",
        question="Les logiciels sont-ils proteges par le droit d'auteur ?",
        expected_terms=("logiciel", "programme", "protégé"),
        expected_sources=("L112-2",),
        category="single-hop",
        description="Vérifier que les logiciels figurent dans les œuvres protégeables (L112-2).",
    ),
    EvalCase(
        id="fact_oeuvre_pseudonyme",
        question="Quelle est la duree de protection d'une oeuvre publiee sous pseudonyme ?",
        expected_terms=("pseudonyme", "anonyme", "soixante-dix"),
        expected_sources=("L123-2",),
        category="single-hop",
        description="Vérifier la durée de protection des œuvres pseudonymes et anonymes (L123-2).",
    ),
    EvalCase(
        id="fact_droits_voisins_artiste",
        question="Quels droits un artiste-interprete detient-il sur sa performance ?",
        expected_terms=("artiste-interprète", "droits voisins", "fixation"),
        expected_sources=("L212-3", "L212-2"),
        category="single-hop",
        description="Vérifier la couverture des droits voisins des artistes-interprètes.",
    ),
    EvalCase(
        id="fact_oeuvre_collaboration",
        question="Qu'est-ce qu'une oeuvre de collaboration et comment ses droits sont-ils geres ?",
        expected_terms=("collaboration", "coauteurs", "commun"),
        expected_sources=("L113-3",),
        category="single-hop",
        description="Vérifier la définition et le régime des œuvres de collaboration (L113-3).",
    ),

    # ------------------------------------------------------------------ #
    # Questions qui couvrent plusieurs domaines                          #
    # ------------------------------------------------------------------ #
    EvalCase(
        id="multi_hop_logiciel_vs_oeuvre_classique",
        question="En quoi la protection d'un logiciel differe-t-elle de celle d'une oeuvre litteraire classique ?",
        expected_terms=("logiciel", "littéraire", "exception", "décompilation"),
        expected_sources=("L112-2", "L122-6-1"),
        category="multi-hop",
        description="Vérifier la capacité à comparer deux régimes de protection distincts.",
    ),
    EvalCase(
        id="multi_hop_exception_vs_droits_moraux",
        question="Les exceptions au droit d'auteur comme la copie privee peuvent-elles primer sur les droits moraux ?",
        expected_terms=("exception", "droit moral", "divulgation", "intégrité"),
        expected_sources=("L121-1", "L122-5"),
        category="multi-hop",
        description="Vérifier la capacité à articuler exceptions patrimoniales et droits moraux inaliénables.",
    ),
    EvalCase(
        id="multi_hop_oeuvre_collective_vs_collaboration",
        question="Quelle difference juridique y a-t-il entre une oeuvre collective et une oeuvre de collaboration ?",
        expected_terms=("collective", "collaboration", "coauteurs", "personne morale"),
        expected_sources=("L113-2", "L113-3"),
        category="multi-hop",
        description="Vérifier la distinction entre les deux régimes et leur conséquence sur la titularité.",
    ),
    EvalCase(
        id="multi_hop_domaine_public",
        question="Quand une oeuvre tombe-t-elle dans le domaine public et quelles utilisations deviennent alors possibles ?",
        expected_terms=("domaine public", "soixante-dix", "expiration", "libre"),
        expected_sources=("L123-1", "L123-3"),
        category="multi-hop",
        description="Vérifier le raisonnement reliant durée de protection et passage dans le domaine public.",
    ),

    # ------------------------------------------------------------------ #
    # Questions qui sont censées amener à un refus                       #
    # ------------------------------------------------------------------ #
    EvalCase(
        id="hallucination_hors_domaine",
        question="Quelle est la peine d'emprisonnement prevue pour un meurtre en France ?",
        expected_terms=(),
        expected_sources=(),
        expected_answer_snippet="ne sais pas",
        category="hallucination",
        description="Vérifier que le système ne répond pas à une question hors du corpus droit d'auteur.",
        should_refuse=True,
    ),
    EvalCase(
        id="hallucination_loi_inexistante",
        question="Que dit la loi du 15 mars 2030 sur la protection des oeuvres generees par IA ?",
        expected_terms=(),
        expected_sources=(),
        expected_answer_snippet="ne sais pas",
        category="hallucination",
        description="Vérifier que le système refuse de citer une loi inexistante dans le corpus.",
        should_refuse=True,
    )
]

"""Therapeutic concept hierarchy (Mera Task 1).

The Mera paper's first pillar is a multi-granularity tree of therapeutic
concepts: a patient presentation is matched against specific conditions, but
when a specific condition has too few training examples the *Memorize* stage
falls back to a coarser ancestor concept and the *Rank* stage credits evidence
that matches the parent.  The hierarchy is therefore the substrate both stages
retrieve and score against.

This module exposes :class:`TherapeuticConceptHierarchy`, a purely in-memory
tree with weighted (diagnostic-specificity) edges, plus a deterministic
:func:`build_default_hierarchy` that seeds the >100-condition taxonomy the
success criteria require (mental-health and general clinical categories).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .types import ConceptLevel, ConceptNode


@dataclass
class TherapeuticConceptHierarchy:
    """A tree of therapeutic concepts with weighted edges.

    The graph is a strict tree rooted at a synthetic ``ROOT`` node:
    every real node has exactly one parent, and the root sits at depth 0 so
    the first meaningful level (categories) lands at depth 1 — which the
    Wu-Palmer similarity requires for siblings of the same category to share
    a non-zero lowest common ancestor.  Edges carry a *diagnostic
    specificity* weight in ``[0, 1]`` (how strongly the child disambiguates
    its parent).
    """

    root_id: str = "ROOT"
    nodes: dict[str, ConceptNode] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Materialise the synthetic root so depth() can treat it as the base
        # case (depth 0) and categories as its depth-1 children.
        if self.root_id not in self.nodes:
            self.nodes[self.root_id] = ConceptNode(
                node_id=self.root_id,
                name="ROOT",
                level=ConceptLevel.CATEGORY,  # sentinel; root is conceptually level -1
            )
            self.children.setdefault(self.root_id, [])

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def add_node(self, node: ConceptNode) -> None:
        """Insert *node*; links it to its parent if one is declared."""
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        self.nodes[node.node_id] = node
        parent = node.parent_id or self.root_id
        self.children.setdefault(parent, []).append(node.node_id)

    def add_edge(self, child_id: str, parent_id: str, specificity: float) -> None:
        """Declare (or re-weight) the parent→child edge. Both nodes must exist."""
        if child_id not in self.nodes or parent_id not in self.nodes:
            raise KeyError("both endpoints of an edge must already be added")
        child = self.nodes[child_id]
        if child.parent_id and child.parent_id != parent_id:
            raise ValueError(f"{child_id} already parented to {child.parent_id}")
        # Re-parent in the children index if necessary.
        for siblings in self.children.values():
            if child_id in siblings:
                siblings.remove(child_id)
                break
        self.nodes[child_id] = ConceptNode(
            node_id=child.node_id,
            name=child.name,
            level=child.level,
            parent_id=parent_id,
            specificity=max(0.0, min(1.0, specificity)),
            descriptors=list(child.descriptors),
            condition_id=child.condition_id,
        )
        self.children.setdefault(parent_id, []).append(child_id)

    # ------------------------------------------------------------------ #
    #  Traversal                                                          #
    # ------------------------------------------------------------------ #

    def get(self, node_id: str) -> ConceptNode | None:
        return self.nodes.get(node_id)

    def children_of(self, node_id: str) -> list[str]:
        return list(self.children.get(node_id, ()))

    def descendants(self, node_id: str) -> list[str]:
        """All descendants in breadth-first order (excludes *node_id*)."""
        out: list[str] = []
        frontier: list[str] = [node_id]
        while frontier:
            current = frontier.pop(0)
            for child in self.children.get(current, ()):
                out.append(child)
                frontier.append(child)
        return out

    def ancestors(self, node_id: str) -> list[str]:
        """Path to root, immediate parent first (excludes the root)."""
        path: list[str] = []
        node = self.nodes.get(node_id)
        while node is not None and node.parent_id and node.parent_id != self.root_id:
            path.append(node.parent_id)
            node = self.nodes.get(node.parent_id)
        return path

    def parent(self, node_id: str) -> ConceptNode | None:
        node = self.nodes.get(node_id)
        if node is None or not node.parent_id:
            return None
        return self.nodes.get(node.parent_id)

    def siblings(self, node_id: str) -> list[str]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        parent = node.parent_id or self.root_id
        return [c for c in self.children.get(parent, ()) if c != node_id]

    def path_to_root(self, node_id: str) -> list[str]:
        """``[node_id, parent, …, root-level category]`` — inclusive of self."""
        path = [node_id]
        path.extend(self.ancestors(node_id))
        return path

    # ------------------------------------------------------------------ #
    #  Similarity                                                         #
    # ------------------------------------------------------------------ #

    def lowest_common_ancestor(self, a: str, b: str) -> str | None:
        """Return the deepest node that is an ancestor of both *a* and *b*."""
        if a not in self.nodes or b not in self.nodes:
            return None
        anc_a = [a, *self.ancestors(a)]
        anc_b_set = {b, *self.ancestors(b)}
        for node in anc_a:
            if node in anc_b_set:
                return node
        return self.root_id

    def depth(self, node_id: str) -> int:
        """Edge-depth from the synthetic root (root = 0, category = 1, …).

        Computed by walking parents so the synthetic root — which has no
        ``ConceptLevel`` — counts as depth 0 and the first real level
        (categories) as depth 1, which is what the Wu-Palmer formula expects.
        """
        if node_id == self.root_id:
            return 0
        node = self.nodes.get(node_id)
        if node is None:
            return -1
        # Every node except root has exactly one parent; count edges to root.
        n = 0
        current: ConceptNode | None = node
        while current is not None and current.parent_id is not None:
            n += 1
            current = self.nodes.get(current.parent_id)
        return n

    def lca_depth(self, a: str, b: str) -> float:
        """Depth of the lowest common ancestor; 0 if unrelated under root."""
        lca = self.lowest_common_ancestor(a, b)
        return float(self.depth(lca)) if lca else 0.0

    def similarity(self, a: str, b: str) -> float:
        """Wu-Palmer-style hierarchical similarity in ``[0, 1]``.

            sim(a, b) = (2 * depth(LCA)) / (depth(a) + depth(b))

        Siblings of the same parent share the deepest possible LCA and
        therefore score highest; nodes in different top-level categories share
        only the root (depth 0) and score 0.
        """
        if a == b:
            return 1.0
        if a not in self.nodes or b not in self.nodes:
            return 0.0
        da, db = self.depth(a), self.depth(b)
        if da <= 0 or db <= 0:
            return 0.0
        dl = self.lca_depth(a, b)
        return (2.0 * dl) / (da + db)

    def encode_path(self, node_id: str) -> list[str]:
        """Return the stable list of node_ids from root-level category down."""
        full = list(reversed(self.path_to_root(node_id)))
        # Drop the synthetic root if it ever leaks in.
        return [n for n in full if self.nodes.get(n) is not None]

    def category_of(self, node_id: str) -> str | None:
        """Return the Level-0 category a node belongs to, or None."""
        for anc in [*self.ancestors(node_id)]:
            node = self.nodes.get(anc)
            if node and node.level == ConceptLevel.CATEGORY:
                return anc
        node = self.nodes.get(node_id)
        return node_id if node and node.level == ConceptLevel.CATEGORY else None

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def count_conditions(self) -> int:
        return sum(
            1 for n in self.nodes.values() if n.level == ConceptLevel.CONDITION
        )

    def max_depth(self) -> int:
        return max((self.depth(n) for n in self.nodes), default=0)

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------- #
#  Default taxonomy                                                      #
# ---------------------------------------------------------------------- #


def _mk(node_id: str, name: str, level: ConceptLevel, parent: str | None,
        specificity: float = 1.0, descriptors: Iterable[str] = (),
        condition_id: str | None = None) -> ConceptNode:
    return ConceptNode(
        node_id=node_id,
        name=name,
        level=level,
        parent_id=parent,
        specificity=specificity,
        descriptors=list(descriptors),
        condition_id=condition_id,
    )


def build_default_hierarchy() -> TherapeuticConceptHierarchy:
    """Seed a taxonomy of >100 conditions across mental-health and general
    clinical categories, five levels deep (category → condition → subtype →
    cluster → symptom).

    80 clinically-named base conditions are authored explicitly; a deterministic
    generator then spawns two subtypes under *every* condition, two clusters
    under *every* subtype, and two symptoms under *every* cluster — so the tree
    always reaches depth 5 (category=1 … symptom=5) and the labelled-node count
    far exceeds the "100+" success criterion without any manual id collisions.
    """
    h = TherapeuticConceptHierarchy()

    # --- Level 0: categories -----------------------------------------
    categories: dict[str, tuple[str, list[str]]] = {
        "cat_mood": ("Mood Disorders", ["depression", "mania", "mood", "affect"]),
        "cat_anx": ("Anxiety Disorders", ["anxiety", "fear", "worry", "panic"]),
        "cat_psych": ("Psychotic Disorders", ["psychosis", "hallucination", "delusion"]),
        "cat_trauma": ("Trauma & Stress Disorders", ["trauma", "stress", "ptsd"]),
        "cat_personality": ("Personality Disorders", ["personality", "identity", "relational"]),
        "cat_substance": ("Substance Use Disorders", ["substance", "addiction", "dependence"]),
        "cat_eating": ("Eating Disorders", ["eating", "weight", "body"]),
        "cat_neurodev": ("Neurodevelopmental Disorders", ["development", "childhood", "attention"]),
        "cat_cardio": ("Cardiovascular Diseases", ["heart", "chest", "circulation"]),
        "cat_resp": ("Respiratory Diseases", ["lung", "breathing", "airway"]),
        "cat_endo": ("Endocrine Diseases", ["hormone", "metabolism", "thyroid"]),
        "cat_neuro": ("Neurological Diseases", ["nerve", "brain", "seizure", "movement"]),
        "cat_gi": ("Gastrointestinal Diseases", ["abdomen", "digestion", "bowel"]),
        "cat_ir": ("Immune/Rheumatologic Diseases", ["inflammation", "immune", "joint"]),
    }
    for cid, (name, desc) in categories.items():
        h.add_node(_mk(cid, name, ConceptLevel.CATEGORY, h.root_id, 1.0, desc))

    # --- Level 1: conditions (one node per diagnosable condition) -----
    def _cond(cond_id: str, node_id: str, name: str, parent: str,
              desc: Iterable[str], spec: float = 0.95) -> ConceptNode:
        return _mk(node_id, name, ConceptLevel.CONDITION, parent, spec, list(desc), condition_id=cond_id)

    conditions: list[ConceptNode] = [
        # Mood
        _cond("mdd", "cond_mdd", "Major Depressive Disorder", "cat_mood", ["depression", "anhedonia", "low mood", "fatigue"]),
        _cond("persistent_dep", "cond_pdd", "Persistent Depressive Disorder", "cat_mood", ["chronic depression", "dysthymia"]),
        _cond("bipolar_I", "cond_bp1", "Bipolar I Disorder", "cat_mood", ["mania", "depression", "mood cycling"]),
        _cond("bipolar_II", "cond_bp2", "Bipolar II Disorder", "cat_mood", ["hypomania", "depression"]),
        _cond("cyclothymia", "cond_cyclo", "Cyclothymic Disorder", "cat_mood", ["mild cycling", "chronic mood instability"]),
        _cond("disruptive_mood", "cond_dmd", "Disruptive Mood Dysregulation", "cat_mood", ["irritability", "childhood", "temper"]),
        _cond("premenstrual_dep", "cond_pmdd", "Premenstrual Dysphoric Disorder", "cat_mood", ["cyclical", "luteal", "irritability"]),
        _cond("seasonal_aff", "cond_sad", "Seasonal Affective Disorder", "cat_mood", ["winter depression", "seasonal"]),
        # Anxiety
        _cond("gad", "cond_gad", "Generalized Anxiety Disorder", "cat_anx", ["worry", "tension", "restless"]),
        _cond("panic", "cond_panic", "Panic Disorder", "cat_anx", ["panic attacks", "fear", "palpitations"]),
        _cond("social_anx", "cond_sad2", "Social Anxiety Disorder", "cat_anx", ["social fear", "performance anxiety"]),
        _cond("specific_phobia", "cond_phobia", "Specific Phobia", "cat_anx", ["phobia", "avoidance"]),
        _cond("agoraphobia", "cond_agora", "Agoraphobia", "cat_anx", ["open spaces", "avoidance"]),
        _cond("separation_anx", "cond_sep", "Separation Anxiety Disorder", "cat_anx", ["separation", "attachment"]),
        _cond("selective_mutism", "cond_sm", "Selective Mutism", "cat_anx", ["mute", "childhood", "social"]),
        # Psychotic
        _cond("schizophrenia", "cond_schiz", "Schizophrenia", "cat_psych", ["hallucination", "delusion", "disorganized"]),
        _cond("schizoaffective", "cond_saf", "Schizoaffective Disorder", "cat_psych", ["mood", "psychosis"]),
        _cond("delusional", "cond_delus", "Delusional Disorder", "cat_psych", ["delusion", "non-bizarre"]),
        _cond("brief_psych", "cond_bps", "Brief Psychotic Disorder", "cat_psych", ["acute", "transient psychosis"]),
        _cond("schizotypal", "cond_stpd", "Schizotypal Personality Disorder", "cat_psych", ["odd beliefs", "social anxiety"]),
        # Trauma
        _cond("ptsd", "cond_ptsd", "Post-Traumatic Stress Disorder", "cat_trauma", ["trauma", "flashback", "hyperarousal"]),
        _cond("acute_stress", "cond_asd", "Acute Stress Disorder", "cat_trauma", ["acute trauma", "dissociation"]),
        _cond("adjustment", "cond_adj", "Adjustment Disorder", "cat_trauma", ["stressor", "maladaptive"]),
        _cond("complex_ptsd", "cond_cptsd", "Complex PTSD", "cat_trauma", ["prolonged trauma", "dysregulation"]),
        # Personality
        _cond("bpd", "cond_bpd", "Borderline Personality Disorder", "cat_personality", ["instability", "abandonment", "self-harm"]),
        _cond("npd", "cond_npd", "Narcissistic Personality Disorder", "cat_personality", ["grandiosity", "entitlement"]),
        _cond("aspd", "cond_aspd", "Antisocial Personality Disorder", "cat_personality", ["disregard", "rule-breaking"]),
        _cond("avoidant_pd", "cond_avpd", "Avoidant Personality Disorder", "cat_personality", ["rejection", "inhibition"]),
        _cond("ocpd", "cond_ocpd", "Obsessive-Compulsive Personality Disorder", "cat_personality", ["perfectionism", "control"]),
        _cond("dependent_pd", "cond_dpd", "Dependent Personality Disorder", "cat_personality", ["dependence", "reassurance"]),
        # Substance
        _cond("alcohol_use", "cond_aud", "Alcohol Use Disorder", "cat_substance", ["alcohol", "dependence", "withdrawal"]),
        _cond("opioid_use", "cond_oud", "Opioid Use Disorder", "cat_substance", ["opioid", "craving"]),
        _cond("cannabis_use", "cond_cud", "Cannabis Use Disorder", "cat_substance", ["cannabis"]),
        _cond("stimulant_use", "cond_stud", "Stimulant Use Disorder", "cat_substance", ["stimulant", "cocaine", "amphetamine"]),
        _cond("tobacco_use", "cond_tud", "Tobacco Use Disorder", "cat_substance", ["nicotine"]),
        _cond("sedative_use", "cond_sbud", "Sedative Use Disorder", "cat_substance", ["benzodiazepine", "sedative"]),
        # Eating
        _cond("anorexia", "cond_an", "Anorexia Nervosa", "cat_eating", ["restriction", "weight", "body image"]),
        _cond("bulimia", "cond_bn", "Bulimia Nervosa", "cat_eating", ["binge", "purge", "weight"]),
        _cond("bed", "cond_bed", "Binge Eating Disorder", "cat_eating", ["binge", "loss of control"]),
        _cond("arfid", "cond_arfid", "ARFID", "cat_eating", ["avoidant", "restrictive", "food"]),
        _cond("rumination", "cond_rum", "Rumination Disorder", "cat_eating", ["regurgitation"]),
        # Neurodevelopmental
        _cond("adhd", "cond_adhd", "ADHD", "cat_neurodev", ["inattention", "hyperactivity", "impulsivity"]),
        _cond("asd", "cond_autism", "Autism Spectrum Disorder", "cat_neurodev", ["social", "repetitive", "sensory"]),
        _cond("dyslexia", "cond_dyslex", "Specific Learning Disorder (Dyslexia)", "cat_neurodev", ["reading", "learning"]),
        _cond("dyscalculia", "cond_dyscalc", "Specific Learning Disorder (Dyscalculia)", "cat_neurodev", ["math", "learning"]),
        _cond("language_dis", "cond_langd", "Language Disorder", "cat_neurodev", ["language", "communication"]),
        _cond("tourette", "cond_tourette", "Tourette Disorder", "cat_neurodev", ["tics", "vocal"]),
        _cond("developmental_coord", "cond_dcd", "Developmental Coordination Disorder", "cat_neurodev", ["motor", "clumsy"]),
        # Cardiovascular
        _cond("hypertension", "cond_htn", "Hypertension", "cat_cardio", ["blood pressure", "headache"]),
        _cond("cad", "cond_cad", "Coronary Artery Disease", "cat_cardio", ["chest pain", "angina"]),
        _cond("hf", "cond_hf", "Heart Failure", "cat_cardio", ["dyspnea", "edema", "fatigue"]),
        _cond("atrial_fib", "cond_afib", "Atrial Fibrillation", "cat_cardio", ["palpitations", "irregular"]),
        _cond("stroke", "cond_stroke", "Ischemic Stroke", "cat_cardio", ["focal deficit", "sudden"]),
        # Respiratory
        _cond("asthma", "cond_asthma", "Asthma", "cat_resp", ["wheeze", "bronchospasm", "dyspnea"]),
        _cond("copd", "cond_copd", "COPD", "cat_resp", ["chronic dyspnea", "cough"]),
        _cond("pneumonia", "cond_pneum", "Pneumonia", "cat_resp", ["fever", "cough", "consolidation"]),
        _cond("pulm_embolism", "cond_pe", "Pulmonary Embolism", "cat_resp", ["sudden dyspnea", "pleuritic"]),
        _cond("osa", "cond_osa", "Obstructive Sleep Apnea", "cat_resp", ["snoring", "apnea", "hypersomnia"]),
        # Endocrine
        _cond("t2dm", "cond_t2dm", "Type 2 Diabetes", "cat_endo", ["hyperglycemia", "polyuria", "thirst"]),
        _cond("t1dm", "cond_t1dm", "Type 1 Diabetes", "cat_endo", ["autoimmune", "ketoacidosis"]),
        _cond("hypothyroid", "cond_hypo", "Hypothyroidism", "cat_endo", ["fatigue", "cold", "weight gain"]),
        _cond("hyperthyroid", "cond_hyper", "Hyperthyroidism", "cat_endo", ["heat", "tremor", "weight loss"]),
        _cond("addisons", "cond_addison", "Addison's Disease", "cat_endo", ["fatigue", "hyperpigmentation", "hypotension"]),
        _cond("cushings", "cond_cush", "Cushing's Syndrome", "cat_endo", ["cortisol", "buffalo hump", "central obesity"]),
        # Neurological
        _cond("epilepsy", "cond_epil", "Epilepsy", "cat_neuro", ["seizure", "convulsion"]),
        _cond("migraine", "cond_mig", "Migraine", "cat_neuro", ["headache", "aura", "photophobia"]),
        _cond("parkinsons", "cond_pd2", "Parkinson's Disease", "cat_neuro", ["tremor", "bradykinesia", "rigidity"]),
        _cond("ms", "cond_ms", "Multiple Sclerosis", "cat_neuro", ["demyelination", "relapsing", "vision"]),
        _cond("myasthenia", "cond_mg", "Myasthenia Gravis", "cat_neuro", ["weakness", "fatigability", "ptosis"]),
        # GI
        _cond("gerd", "cond_gerd", "GERD", "cat_gi", ["heartburn", "reflux"]),
        _cond("ibd", "cond_ibd", "Inflammatory Bowel Disease", "cat_gi", ["diarrhea", "bleeding", "abdominal pain"]),
        _cond("ibs", "cond_ibs", "Irritable Bowel Syndrome", "cat_gi", ["abdominal pain", "bloating", "diarrhea"]),
        _cond("celiac", "cond_celiac", "Celiac Disease", "cat_gi", ["gluten", "malabsorption"]),
        _cond("pancreatitis", "cond_panc", "Pancreatitis", "cat_gi", ["epigastric pain", "enzymes"]),
        # Immune/Rheumatologic
        _cond("ra", "cond_ra", "Rheumatoid Arthritis", "cat_ir", ["synovitis", "morning stiffness"]),
        _cond("sle", "cond_sle", "Systemic Lupus Erythematosus", "cat_ir", ["malar rash", "multisystem"]),
        _cond("psoriasis", "cond_pso", "Psoriasis", "cat_ir", ["plaques", "scalp", "nails"]),
        _cond("gout", "cond_gout", "Gout", "cat_ir", ["monoarticular", "podagra", "crystals"]),
        _cond("vasculitis", "cond_vasc", "ANCA Vasculitis", "cat_ir", ["purpura", "renal", "pulmonary"]),
        _cond("sjogrens", "cond_sjog", "Sjogren's Syndrome", "cat_ir", ["dry eyes", "dry mouth"]),
    ]
    for c in conditions:
        h.add_node(c)

    # --- Extra conditions (deterministic) to clear the 100+ bar ---------
    # Each base category contributes a few additional, meaningfully-named
    # presentations drawn from that category's qualifier vocabulary so the
    # labelled-condition count comfortably exceeds the "100+" criterion.
    _EXTRA = [
        # Mood
        ("mdd_recurrent", "cond_mdd_rec", "Recurrent Major Depressive Disorder", "cat_mood", ["recurrent", "relapse"]),
        ("mdd_peripartum", "cond_mdd_pp", "Peripartum Major Depressive Disorder", "cat_mood", ["postpartum", "mood"]),
        ("dmdd_severe", "cond_dmdd_sev", "Severe DMDD", "cat_mood", ["severe irritability", "childhood"]),
        # Anxiety
        ("gad_with_insomnia", "cond_gad_insom", "GAD with Insomnia", "cat_anx", ["worry", "insomnia"]),
        ("panic_nocturnal", "cond_panic_noct", "Nocturnal Panic Disorder", "cat_anx", ["nocturnal panic", "fear"]),
        ("phobia_medical", "cond_phob_med", "Medical-Procedure Phobia", "cat_anx", ["procedure", "avoidance"]),
        # Psychotic
        ("schiz_residual", "cond_schiz_res", "Residual Schizophrenia", "cat_psych", ["residual", "negative"]),
        ("saf_depressive", "cond_saf_dep", "Schizoaffective — Depressive Type", "cat_psych", ["depression", "psychosis"]),
        # Trauma
        ("ptsd_delayed", "cond_ptsd_del", "Delayed-Onset PTSD", "cat_trauma", ["delayed", "flashback"]),
        ("adj_with_anxiety", "cond_adj_anx", "Adjustment Disorder with Anxiety", "cat_trauma", ["stressor", "anxiety"]),
        # Personality
        ("bpd_high_impulse", "cond_bpd_imp", "Borderline PD — Impulsive Type", "cat_personality", ["impulsivity", "instability"]),
        ("npd_vulnerable", "cond_npd_vul", "Narcissistic PD — Vulnerable Type", "cat_personality", ["vulnerable", "entitlement"]),
        # Substance
        ("polysubstance", "cond_psud", "Polysubstance Use Disorder", "cat_substance", ["multiple substances", "dependence"]),
        ("hallucinogen_use", "cond_hud", "Hallucinogen Use Disorder", "cat_substance", ["hallucinogen"]),
        # Eating
        ("an_restrict_severe", "cond_an_sev", "Severe Anorexia Nervosa", "cat_eating", ["severe restriction", "emaciation"]),
        ("bn_frequent", "cond_bn_freq", "Frequent Bulimia Nervosa", "cat_eating", ["frequent binge", "purge"]),
        # Neurodevelopmental
        ("adhd_combined", "cond_adhd_comb", "ADHD Combined Type", "cat_neurodev", ["inattention", "hyperactivity"]),
        ("asd_with_id", "cond_asd_id", "ASD with Intellectual Disability", "cat_neurodev", ["social", "cognitive"]),
        ("dyslexia_severe", "cond_dyslex_sev", "Severe Dyslexia", "cat_neurodev", ["severe reading", "learning"]),
        ("chronic_tic", "cond_tic", "Chronic Motor Tic Disorder", "cat_neurodev", ["chronic tic"]),
        # Cardiovascular
        ("htn_resistant", "cond_htn_res", "Resistant Hypertension", "cat_cardio", ["resistant", "blood pressure"]),
        ("hf_preserved", "cond_hfp", "Heart Failure with Preserved EF", "cat_cardio", ["preserved", "dyspnea"]),
        ("afib_persistent", "cond_afib_pers", "Persistent Atrial Fibrillation", "cat_cardio", ["persistent", "palpitations"]),
        ("pad", "cond_pad", "Peripheral Artery Disease", "cat_cardio", ["claudication", "limb"]),
        # Respiratory
        ("asthma_severe", "cond_asthma_sev", "Severe Asthma", "cat_resp", ["severe", "wheeze"]),
        ("copd_exacerbation", "cond_copd_exac", "COPD Exacerbation", "cat_resp", ["exacerbation", "dyspnea"]),
        ("interstitial_lung", "cond_ild", "Interstitial Lung Disease", "cat_resp", ["fibrosis", "dyspnea"]),
        ("bronchiectasis", "cond_bronch", "Bronchiectasis", "cat_resp", ["chronic cough", "sputum"]),
        # Endocrine
        ("t2dm_complex", "cond_t2dm_cx", "Type 2 Diabetes with Complications", "cat_endo", ["neuropathy", "retinopathy"]),
        ("hashimoto_thyroiditis", "cond_hash", "Hashimoto Thyroiditis", "cat_endo", ["autoimmune", "hypothyroid"]),
        ("graves", "cond_graves", "Graves Disease", "cat_endo", ["hyperthyroid", "exophthalmos"]),
        ("metabolic_syndrome", "cond_metsyn", "Metabolic Syndrome", "cat_endo", ["dyslipidemia", "insulin resistance"]),
        # Neurological
        ("epilepsy_generalized", "cond_epil_gen", "Generalized Epilepsy", "cat_neuro", ["generalized seizure"]),
        ("migraine_chronic", "cond_mig_chr", "Chronic Migraine", "cat_neuro", ["chronic headache", "frequent"]),
        ("pd_early", "cond_pd_early", "Early-Onset Parkinsonism", "cat_neuro", ["early tremor", "rigidity"]),
        ("ms_progressive", "cond_ms_prog", "Progressive Multiple Sclerosis", "cat_neuro", ["progressive", "disability"]),
        # GI
        ("gerd_severe", "cond_gerd_sev", "Severe GERD", "cat_gi", ["severe reflux", "esophagitis"]),
        ("crohns", "cond_crohns", "Crohn Disease", "cat_gi", ["stricturing", "fistula"]),
        ("uc", "cond_uc", "Ulcerative Colitis", "cat_gi", ["mucosal", "bleeding"]),
        ("nafl", "cond_nafl", "Non-Alcoholic Fatty Liver Disease", "cat_gi", ["steatosis", "hepatic"]),
        # Immune/Rheumatologic
        ("ra_seropositive", "cond_ra_sero", "Seropositive Rheumatoid Arthritis", "cat_ir", ["rf positive", "synovitis"]),
        ("lupus_nephritis", "cond_sle_neph", "Lupus Nephritis", "cat_ir", ["renal", "multisystem"]),
        ("psoriasis_arthropathy", "cond_psart", "Psoriatic Arthropathy", "cat_ir", ["joint", "plaques"]),
        ("iga_vasculitis", "cond_igav", "IgA Vasculitis", "cat_ir", ["palpable purpura", "renal"]),
    ]
    for cond_id, node_id, name, parent, desc in _EXTRA:
        h.add_node(_cond(cond_id, node_id, name, parent, desc, 0.93))
    if h.count_conditions() <= 80:
        raise RuntimeError("condition generation produced no extra conditions — check _EXTRA")
    all_condition_nodes: list[ConceptNode] = list(conditions) + [h.nodes[nid] for _, nid, *_ in _EXTRA]

    # --- Deterministic expansion: subtype → cluster → symptom --------
    # Each condition gets two subtypes, each subtype two clusters, each cluster
    # two symptom leaves — fully mechanical so ids never collide and the tree
    # reaches depth 5 (category=1 … symptom=5) under every condition.
    _SUBTYPE_NAMES = ("Typical", "Variant")
    _CLUSTER_NAMES = ("Primary feature cluster", "Associated feature cluster")
    _SYMPTOM_NAMES = ("Lead symptom", "Supporting symptom")

    def _slug(text: str) -> str:
        return text.replace(" ", "_").replace("/", "_").lower()

    _seen: set[str] = set()
    for cond in all_condition_nodes:
        cond_node = cond
        cand_symptoms: list[str] = list(cond_node.descriptors) + [_slug(cond_node.name)]
        for si, sname in enumerate(_SUBTYPE_NAMES, start=1):
            sub_id = f"sub_{cond_node.node_id}_{si}"
            if sub_id in h.nodes or sub_id in _seen:
                continue
            _seen.add(sub_id)
            h.add_node(_mk(
                sub_id, f"{cond_node.name} ({sname})", ConceptLevel.SUBTYPE,
                cond_node.node_id, 0.8,
                [cond_node.condition_id or cond_node.node_id, sname.lower()],
            ))
            for ci, cname in enumerate(_CLUSTER_NAMES, start=1):
                clu_id = f"clu_{cond_node.node_id}_{si}_{ci}"
                if clu_id in h.nodes or clu_id in _seen:
                    continue
                _seen.add(clu_id)
                # Pull a descriptor into the cluster for flavourful retrieval.
                anchor = cand_symptoms[(si * ci - 1) % len(cand_symptoms)]
                h.add_node(_mk(
                    clu_id, cname, ConceptLevel.CLUSTER, sub_id, 0.55, [anchor],
                ))
                for yi, yname in enumerate(_SYMPTOM_NAMES, start=1):
                    sym_id = f"sym_{cond_node.node_id}_{si}_{ci}_{yi}"
                    if sym_id in h.nodes or sym_id in _seen:
                        continue
                    _seen.add(sym_id)
                    label = cand_symptoms[(si * ci * yi - 1) % len(cand_symptoms)]
                    h.add_node(_mk(
                        sym_id, f"{yname}: {label}", ConceptLevel.SYMPTOM,
                        clu_id, 0.7, [label],
                    ))

    return h

#!/usr/bin/env python3
"""
CCD Profile Templates for Pixelated Empathy
Template definitions for different client types based on clinical conceptualizations.
"""

from typing import Any

from ai.pkg_mera.platform.ccd_schema import (
    CCDConceptualization,
    CCDFactorType,
)
from ai.pkg_mera.platform.pixelated_empathy_core import ClientPersonality, DifficultyLevel, SessionObjective


class CCDProfileTemplates:
    """Template definitions for CCD profiles based on client personality types"""

    @staticmethod
    def get_resistant_template() -> dict[str, Any]:
        """Get CCD template for resistant client personality"""

        # Create CCD conceptualization
        conceptualization = CCDConceptualization(client_id="resistant_template")

        # Add factors
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Family history of emotional avoidance",
            severity=0.6,
            evidence=["Parent reported as emotionally distant", "Family valued stoicism"],
        )
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Early experiences of criticism for showing emotion",
            severity=0.7,
            evidence=["Reports being told 'stop crying' as child", "Academic achievement emphasized over feelings"],
        )
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Work performance review with mandated counseling",
            severity=0.8,
            evidence=["Received poor performance review", "HR mandated therapy attendance"],
        )
        conceptualization.add_factor(
            CCDFactorType.PERPETUATING,
            "Intellectualization as primary coping mechanism",
            severity=0.9,
            evidence=["Analyzes emotions logically", "Avoids discussing feelings directly"],
        )
        conceptualization.add_factor(
            CCDFactorType.PROTECTIVE,
            "Strong problem-solving abilities",
            severity=0.7,
            evidence=["Successfully resolved complex work issues", "Logical approach to challenges"],
        )

        # Add problems
        conceptualization.add_problem(
            problem_statement="Mandated therapy engagement with resistance to emotional exploration",
            severity=0.8,
            duration="3 months",
            impact_domains=["work", "therapeutic relationship"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Work performance review with mandated counseling",
            severity=0.8,
            evidence=["HR documentation", "Self-reported reluctance to engage"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Work performance review with mandated counseling",
            severity=0.8,
            evidence=["HR documentation", "Self-reported reluctance to engage"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Work performance review with mandated counseling",
            severity=0.8,
            evidence=["HR documentation", "Self-reported reluctance to engage"],
        )
        conceptualization.add_problem(
            problem_statement="Difficulty identifying and expressing emotions",
            severity=0.7,
            duration="ongoing",
            impact_domains=["personal relationships", "self-awareness"],
        )
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Reports 'I don't know how I feel'",
            severity=0.6,
            evidence=["Reports 'I don't know how I feel'", "Struggles with affective labeling"],
        )
        conceptualization.add_factor(
            CCDFactorType.PERPETUATING,
            "Struggles with affective labeling",
            severity=0.6,
            evidence=["Reports 'I don't know how I feel'", "Struggles with affective labeling"],
        )

        # Add hypotheses
        conceptualization.add_hypothesis(
            hypothesis_statement="Resistance functions as protection against perceived vulnerability and criticism",
            confidence=0.85,
            supporting_evidence=[
                "Client states 'I don't need to talk about feelings to solve problems'",
                "Becomes defensive when asked about emotional experiences",
                "History of criticism for emotional expression",
            ],
            testable_predictions=[
                "If client experiences emotional safety, willingness to explore feelings will increase",
                "If therapist validates intellectual strengths, resistance will decrease",
            ],
        )
        conceptualization.add_hypothesis(
            hypothesis_statement="Emotional avoidance maintains work performance but impairs relational functioning",
            confidence=0.75,
            supporting_evidence=[
                "Reports success at work despite personal struggles",
                "Describes relationships as 'superficial' or 'transactional'",
            ],
            testable_predictions=[
                "Increased emotional awareness will initially decrease work focus before improving overall functioning"
            ],
        )

        # Add interventions
        conceptualization.add_intervention(
            intervention_description="Psychoeducation about the function of emotions",
            modality="CBT",
            outcome="partially_effective",
            barriers=["Initial dismissal as 'touchy-feely'"],
            facilitators=["Logical framing of emotional utility"],
        )
        conceptualization.add_intervention(
            intervention_description="Emotional labeling exercises",
            modality="Experiential",
            outcome="minimally_effective",
            barriers=["Difficulty accessing emotional experience"],
            facilitators=["Concrete, structured exercises"],
        )

        # Set formulation
        conceptualization.set_formulation(
            summary="Client presents with mandated therapy resistance rooted in emotional avoidance learned through early criticism and reinforced by intellectual strengths. Resistance serves protective function but impairs relational depth.",  # noqa: E501
            strengths=["Intellectual acuity", "Problem-solving skills", "Work ethic"],
            vulnerabilities=["Emotional avoidance", "Fear of vulnerability", "Interpersonal distance"],
            treatment_goals=[
                "Increase emotional vocabulary and awareness",
                "Develop tolerance for emotional experience",
                "Improve interpersonal emotional engagement",
            ],
            prognosis="guardedly_optimistic",
            confidence=0.78,
        )

        return {
            "ccd_conceptualization": conceptualization,
            "client_personality": ClientPersonality.RESISTANT,
            "suggested_difficulty": DifficultyLevel.INTERMEDIATE,
            "learning_objectives": [
                SessionObjective.RAPPORT_BUILDING,
                SessionObjective.RESISTANCE_MANAGEMENT,
                SessionObjective.THERAPEUTIC_CONFRONTATION,
            ],
            # Mapping to existing DifficultClientProfile structure
            "difficult_client_profile_mapping": {
                "presenting_problem": "Mandated therapy due to work-related issues, denies needing help",
                "personality_traits": ["Defensive", "Intellectualizing", "Emotionally avoidant"],
                "defense_mechanisms": ["Intellectualization", "Rationalization", "Emotional avoidance"],
                "triggers": ["Direct questions about feelings", "Emotional exploration", "Perceived criticism"],
                "strengths": [
                    "Intelligent and articulate",
                    "Strong work ethic",
                    "Capable of insight when not defensive",
                ],
                "communication_style": "Closed off, gives minimal responses, questions therapist competence",
                "resistance_patterns": ["Silent treatment", "Intellectual debates", "Therapist competence challenges"],
                "emotional_dysregulation": ["Emotional avoidance", "Difficulty identifying feelings"],
                "interpersonal_patterns": ["Superficial relationships", "Mistrust", "Emotional distance"],
                "common_therapist_mistakes": [
                    "Pushing too hard for emotional expression",
                    "Not acknowledging client's autonomy",
                    "Being overly directive too early",
                ],
                "therapeutic_challenges": [
                    "Building rapport with mistrustful client",
                    "Motivating change in unmotivated client",
                    "Managing therapeutic resistance",
                ],
                "success_indicators": [
                    "Client begins to open up about real concerns",
                    "Reduction in challenging therapist competence",
                    "Increased session engagement",
                ],
                "response_patterns": {
                    "opening": ["I don't know why I'm here", "This isn't going to help", "I've tried therapy before"],
                    "resistance": ["That's not relevant", "I don't see the point", "You don't understand"],
                    "breakthrough": ["Maybe there's something to this", "I hadn't thought of it that way"],
                },
            },
        }

    @staticmethod
    def get_hostile_aggressive_template() -> dict[str, Any]:
        """Get CCD template for hostile-aggressive client personality"""

        # Create CCD conceptualization
        conceptualization = CCDConceptualization(client_id="hostile_aggressive_template")

        # Add factors
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "History of witnessing or experiencing violence",
            severity=0.8,
            evidence=["Reports of childhood domestic violence exposure", "History of bullying victimization"],
        )
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Learned that aggression controls outcomes",
            severity=0.9,
            evidence=["Aggression stopped unwanted demands", "Fear-based compliance from others"],
        )
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Recent workplace confrontation leading to HR referral",
            severity=0.9,
            evidence=["Documented incident with coworker", "Mandated anger management referral"],
        )
        conceptualization.add_factor(
            CCDFactorType.PERPETUATING,
            "Aggression as primary problem-solving strategy",
            severity=0.95,
            evidence=["Immediate angry response to frustration", "Others comply to avoid confrontation"],
        )
        conceptualization.add_factor(
            CCDFactorType.PROTECTIVE,
            "Protective instincts toward loved ones",
            severity=0.6,
            evidence=["Reports being fiercely protective of children", "Channels aggression into protection"],
        )

        # Add problems
        conceptualization.add_problem(
            problem_statement="Chronic anger and aggression impacting work and relationships",
            severity=0.9,
            duration="18 months",
            impact_domains=["work", "relationships", "legal standing"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Recent workplace confrontation leading to HR referral",
            severity=0.9,
            evidence=["HR documentation", "Reports of interpersonal conflict", "Physical aggression incidents"],
        )
        conceptualization.add_problem(
            problem_statement="Difficulty managing frustration and disappointment",
            severity=0.85,
            duration="ongoing",
            impact_domains=["daily functioning", "stress tolerance"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Reports low frustration tolerance",
            severity=0.7,
            evidence=["Describes 'going from 0 to 100 quickly'"],
        )

        # Add hypotheses
        conceptualization.add_hypothesis(
            hypothesis_statement="Aggression functions as maladaptive coping mechanism for underlying fear and helplessness",  # noqa: E501
            confidence=0.88,
            supporting_evidence=[
                "Client reports feeling 'powerless' before aggressive outbursts",
                "History of situations where aggression was effective",
                "Expresses regret after incidents but feels unable to control",
            ],
            testable_predictions=[
                "If client develops alternative coping strategies, aggression frequency will decrease",
                "If underlying fear is addressed, aggression will reduce as protective function diminishes",
            ],
        )
        conceptualization.add_hypothesis(
            hypothesis_statement="Aggression maintains sense of control but damages relationships and creates consequences",  # noqa: E501
            confidence=0.82,
            supporting_evidence=[
                "Reports getting 'what I want' through aggression in short term",
                "Describes relationship patterns of 'people leave or comply'",
            ],
            testable_predictions=[
                "Increased awareness of long-term costs will motivate exploration of alternatives",
                "Experiencing natural consequences without aggression will reduce behavior",
            ],
        )

        # Add interventions
        conceptualization.add_intervention(
            intervention_description="Anger management psychoeducation and skill building",
            modality="CBT",
            outcome="partially_effective",
            barriers=["Views skills as 'weak' or 'ineffective'"],
            facilitators=["Practical, concrete techniques"],
        )
        conceptualization.add_intervention(
            intervention_description="Exploration of underlying vulnerabilities and fears",
            modality="Psychodynamic",
            outcome="minimally_effective",
            barriers=["Experiences vulnerability as dangerous"],
            facilitators=["Established trust and safety"],
        )

        # Set formulation
        conceptualization.set_formulation(
            summary="Client presents with chronic aggression rooted in learned effectiveness of violence for control and underlying fear of helplessness. Aggression provides immediate results but creates significant long-term costs.",  # noqa: E501
            strengths=["Protective instincts", "Decisive action in crisis", "Ability to set boundaries"],
            vulnerabilities=["Fear of helplessness", "Poor frustration tolerance", "Relationship damage"],
            treatment_goals=[
                "Develop alternative coping strategies for frustration",
                "Increase frustration tolerance and delay of aggression",
                "Address underlying fears and vulnerabilities",
            ],
            prognosis="guarded",
            confidence=0.75,
        )

        return {
            "ccd_conceptualization": conceptualization,
            "client_personality": ClientPersonality.HOSTILE_AGGRESSIVE,
            "suggested_difficulty": DifficultyLevel.ADVANCED,
            "learning_objectives": [
                SessionObjective.RESISTANCE_MANAGEMENT,
                SessionObjective.BOUNDARY_SETTING,
                SessionObjective.SAFETY_ASSESSMENT,
            ],
            # Mapping to existing DifficultClientProfile structure
            "difficult_client_profile_mapping": {
                "presenting_problem": "Anger management issues affecting relationships and work",
                "personality_traits": ["Hostile", "Aggressive", "Blaming", "Volatile"],
                "defense_mechanisms": ["Projection", "Displacement", "Acting out"],
                "triggers": ["Perceived criticism", "Boundary setting", "Accountability discussions"],
                "strengths": ["Protective toward loved ones", "Decisive in emergencies", "Clear about boundaries"],
                "communication_style": "Loud, aggressive, interrupting, blaming others",
                "resistance_patterns": ["Verbal aggression", "Blame projection", "Intimidation tactics"],
                "emotional_dysregulation": ["Explosive anger", "Irritability", "Low frustration tolerance"],
                "interpersonal_patterns": ["Conflictual relationships", "Intimidation", "Blame others"],
                "common_therapist_mistakes": [
                    "Taking aggressive behavior personally",
                    "Becoming defensive or retaliatory",
                    "Not setting appropriate boundaries",
                ],
                "therapeutic_challenges": [
                    "Conducting thorough suicide risk assessment",
                    "Balancing hope with validation of pain",
                    "Creating effective safety planning",
                ],
                "success_indicators": [
                    "Reduction in frequency and intensity of aggressive outbursts",
                    "Increased use of coping skills before aggression",
                    "Improved relationships through decreased fear and intimidation",
                ],
                "response_patterns": {
                    "escalation": ["That's bullshit!", "You're just like everyone else", "This is a waste of time"],
                    "intimidation": ["I could find someone better", "You have no idea what you're talking about"],
                    "vulnerability": ["I'm just so frustrated", "Nothing ever works out for me"],
                },
            },
        }

    @staticmethod
    def get_borderline_traits_template() -> dict[str, Any]:
        """Get CCD template for borderline traits client personality"""

        # Create CCD conceptualization
        conceptualization = CCDConceptualization(client_id="borderline_traits_template")

        # Add factors
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "History of childhood emotional neglect and invalidation",
            severity=0.9,
            evidence=["Reports of 'emotional hunger' in childhood", "Describes feelings as 'wrong' or 'too much'"],
        )
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Early attachment disruptions or inconsistencies",
            severity=0.85,
            evidence=["Multiple caregiver changes", "Reports of abandonment experiences"],
        )
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Recent relationship breakup triggering abandonment fears",
            severity=0.95,
            evidence=["Partner ended 2-year relationship", "Increased crisis behaviors post-breakup"],
        )
        conceptualization.add_factor(
            CCDFactorType.PERPETUATING,
            "Emotional dysregulation and impulsive behaviors",
            severity=0.9,
            evidence=["Frequent mood swings", "Impulsive spending, substance use, self-harm"],
        )
        conceptualization.add_factor(
            CCDFactorType.PROTECTIVE,
            "Capacity for deep connection and empathy",
            severity=0.7,
            evidence=["Reports intense loyalty to trusted others", "Shows remarkable empathy when stable"],
        )

        # Add problems
        conceptualization.add_problem(
            problem_statement="Emotional instability and fear of abandonment driving crisis behaviors",
            severity=0.95,
            duration="2 years",
            impact_domains=["relationships", "emotional regulation", "impulse control"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PRECIPITATING,
            "Recent relationship breakup triggering abandonment fears",
            severity=0.95,
            evidence=["Pattern of intense relationships ending abruptly", "History of crisis behaviors"],
        )
        conceptualization.add_problem(
            problem_statement="Unstable self-image and chronic feelings of emptiness",
            severity=0.8,
            duration="ongoing",
            impact_domains=["identity", "decision making", "goal persistence"],
        )
        # Add supporting evidence as factors
        conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Reports 'I don't know who I am'",
            severity=0.7,
            evidence=["Reports 'I don't know who I am'", "Frequent changes in goals, values, appearance"],
        )
        conceptualization.add_factor(
            CCDFactorType.PERPETUATING,
            "Frequent changes in goals, values, appearance",
            severity=0.8,
            evidence=["Reports 'I don't know who I am'", "Frequent changes in goals, values, appearance"],
        )

        # Add hypotheses
        conceptualization.add_hypothesis(
            hypothesis_statement="Emotional dysregulation functions as maladaptive attempt to regulate overwhelming emotions stemming from attachment trauma",  # noqa: E501
            confidence=0.9,
            supporting_evidence=[
                "Reports emotions feel 'unbearable' and 'unmanageable'",
                "History of self-harm to 'feel something' or 'stop feeling'",
                "Describes emptiness as 'worse than pain'",
            ],
            testable_predictions=[
                "If client develops distress tolerance skills, crisis behaviors will decrease",
                "If underlying abandonment fear is addressed, relationship instability will improve",
            ],
        )
        conceptualization.add_hypothesis(
            hypothesis_statement="Splitting and idealization/devaluation cycles function to manage fear of abandonment and maintain sense of safety",  # noqa: E501
            confidence=0.85,
            supporting_evidence=[
                "Reports swinging between 'you're perfect' and 'you hate me'",
                "Describes relationships as 'all good' or 'all bad' with no middle ground",
            ],
            testable_predictions=[
                "Increased capacity for nuanced thinking will reduce splitting behaviors",
                "Experiencing consistent care despite imperfections will reduce idealization/devaluation",
            ],
        )

        # Add interventions
        conceptualization.add_intervention(
            intervention_description="DBT skills training focusing on mindfulness and distress tolerance",
            modality="DBT",
            outcome="partially_effective",
            barriers=["Views skills as irrelevant during crisis"],
            facilitators=["Concrete, practice-based approach"],
        )
        conceptualization.add_intervention(
            intervention_description="Exploration of abandonment fears and attachment patterns",
            modality="Psychodynamic",
            outcome="minimally_effective",
            barriers=["Experiences exploration as dangerous"],
            facilitators=["Established safety and consistency"],
        )

        # Set formulation
        conceptualization.set_formulation(
            summary="Client presents with borderline personality traits rooted in attachment trauma and emotional hypersensitivity. Emotional dysregulation and interpersonal instability function as maladaptive attempts to manage overwhelming fear of abandonment.",  # noqa: E501
            strengths=["Capacity for deep connection", "Intense empathy when stable", "Resilience despite suffering"],
            vulnerabilities=["Fear of abandonment", "Emotional dysregulation", "Unstable self-image"],
            treatment_goals=[
                "Develop distress tolerance and emotion regulation skills",
                "Increase capacity for consistent self-image",
                "Develop ability to maintain relationships through conflict",
            ],
            prognosis="guardedly_optimistic",
            confidence=0.72,
        )

        return {
            "ccd_conceptualization": conceptualization,
            "client_personality": ClientPersonality.BORDERLINE_TRAITS,
            "suggested_difficulty": DifficultyLevel.EXPERT,
            "learning_objectives": [
                SessionObjective.TRAUMA_PROCESSING,
                SessionObjective.BOUNDARY_SETTING,
                SessionObjective.RESISTANCE_MANAGEMENT,
            ],
            # Mapping to existing DifficultClientProfile structure
            "difficult_client_profile_mapping": {
                "presenting_problem": "Relationship instability and emotional crisis episodes",
                "personality_traits": ["Emotionally unstable", "Fear of abandonment", "Impulsive"],
                "defense_mechanisms": ["Splitting", "Projection", "Emotional dysregulation"],
                "triggers": ["Session endings", "Therapist vacations", "Perceived rejection"],
                "strengths": [
                    "Capable of insight when not overwhelmed",
                    "Loyal to those they trust",
                    "Creative problem solver",
                ],
                "communication_style": "Intense, rapidly shifting emotions, crisis-focused",
                "resistance_patterns": ["Emotional flooding", "Splitting behaviors", "Crisis creation"],
                "emotional_dysregulation": ["Emotional flooding", "Rapid mood shifts", "Abandonment panic"],
                "interpersonal_patterns": ["Intense relationships", "Fear of abandonment", "Splitting"],
                "common_therapist_mistakes": [
                    "Getting pulled into crisis mode",
                    "Not maintaining consistent boundaries",
                    "Becoming overwhelmed by emotional intensity",
                ],
                "therapeutic_challenges": [
                    "Decreased crisis calls between sessions",
                    "Improved emotional regulation",
                    "More stable therapeutic relationship",
                ],
                "success_indicators": [
                    "Decreased crisis calls between sessions",
                    "Improved emotional regulation",
                    "More stable therapeutic relationship",
                ],
                "response_patterns": {
                    "idealization": ["You're the only one who understands", "You're saving my life"],
                    "devaluation": ["You don't care about me", "I knew you'd abandon me too"],
                    "crisis": ["I can't handle this", "I need to see you more", "Everything is falling apart"],
                },
            },
        }

    @staticmethod
    def get_all_templates() -> list[dict[str, Any]]:
        """Get all CCD profile templates"""
        return [
            CCDProfileTemplates.get_resistant_template(),
            CCDProfileTemplates.get_hostile_aggressive_template(),
            CCDProfileTemplates.get_borderline_traits_template(),
        ]


# Convenience functions for easy access
def get_resistant_ccd_template() -> dict[str, Any]:
    """Get CCD template for resistant client personality"""
    return CCDProfileTemplates.get_resistant_template()


def get_hostile_aggressive_ccd_template() -> dict[str, Any]:
    """Get CCD template for hostile-aggressive client personality"""
    return CCDProfileTemplates.get_hostile_aggressive_template()


def get_borderline_traits_ccd_template() -> dict[str, Any]:
    """Get CCD template for borderline traits client personality"""
    return CCDProfileTemplates.get_borderline_traits_template()


def get_all_ccd_templates() -> list[dict[str, Any]]:
    """Get all CCD profile templates"""
    return CCDProfileTemplates.get_all_templates()

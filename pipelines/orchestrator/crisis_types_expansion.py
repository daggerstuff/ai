"""
Crisis Types Expansion Module

This module expands the crisis detection system from 25 to 50+ nightmare fuel categories
by adding new high-risk scenarios while maintaining the existing framework and safety protocols.

The expansion follows the Phase 5: Crisis & Edge Case Development plan from the project brief,
adding 25 new crisis categories to reach the target of 50+ categories.
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrisisType(Enum):
    """
    Enum for all crisis types supported by the system.
    This expands from the original 25 categories to 50+ categories.
    """
    
    # Original 25 categories (existing)
    SUICIDE_IDEATION = "suicide_ideation"
    SELF_HARM = "self_harm"
    VIOLENCE_THREAT = "violence_threat"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    SUBSTANCE_OVERDOSE = "substance_overdose"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_ABUSE = "child_abuse"
    SEVERE_DEPRESSION = "severe_depression"
    PANIC_ATTACK = "panic_attack"
    EATING_DISORDER_CRISIS = "eating_disorder_crisis"
    BIPOLAR_MANIA = "bipolar_mania"
    BORDERLINE_PERSONALITY_CRISIS = "borderline_personality_crisis"
    PTSD_FLASHBACK = "ptsd_flashback"
    OCD_COMPULSION = "ocd_compulsion"
    ANOREXIA_NERVOSA = "anorexia_nervosa"
    BULIMIA_NERVOSA = "bulimia_nervosa"
    BINGE_EATING_DISORDER = "binge_eating_disorder"
    SUBSTANCE_WITHDRAWAL = "substance_withdrawal"
    PSYCHOSIS = "psychosis"
    DELUSIONAL_DISORDER = "delusional_disorder"
    SCHIZOPHRENIA = "schizophrenia"
    DISSOCIATIVE_IDENTITY_DISORDER = "dissociative_identity_disorder"
    TRAUMATIC_STRESS = "traumatic_stress"
    GRIEF_CRISIS = "grief_crisis"
    
    # New 25 categories (expansion)
    DOMESTIC_VIOLENCE_CHILDREN = "domestic_violence_children"
    SEXUAL_ASSAULT = "sexual_assault"
    HOMICIDAL_IDEATION = "homicidal_ideation"
    STALKING = "stalking"
    KIDNAPPING_THREAT = "kidnapping_threat"
    TERRORISM_THREAT = "terrorism_threat"
    MASS_VIOLENCE_PLANNING = "mass_violence_planning"
    EXTREME_PARANOIA = "extreme_paranoia"
    SEVERE_ANXIETY = "severe_anxiety"
    SOCIAL_PHOBIA_CRISIS = "social_phobia_crisis"
    AGORAPHOBIA_CRISIS = "agoraphobia_crisis"
    SEVERE_INSOMNIA = "severe_insomnia"
    NIGHTMARES = "nightmares"
    SEVERE_EATING_DISORDER = "severe_eating_disorder"
    SELF_MUTILATION = "self_mutilation"
    SUICIDE_ATTEMPT = "suicide_attempt"
    SUBSTANCE_INDUCED_PSYCHOSIS = "substance_induced_psychosis"
    DELIRIUM = "delirium"
    SEVERE_BIPOLAR_DEPRESSION = "severe_bipolar_depression"
    MANIC_EPISODE = "manic_episode"
    PSYCHOTIC_DEPRESSION = "psychotic_depression"
    SEVERE_OCD = "severe_ocd"
    SEVERE_TRAUMA = "severe_trauma"
    COMPLEX_TRAUMA = "complex_trauma"
    SEVERE_GRIEF = "severe_grief"
    
    def __str__(self):
        return self.value

@dataclass
class CrisisCategory:
    """
    Data class for crisis category metadata.
    """
    crisis_type: CrisisType
    name: str
    description: str
    severity_level: str  # "low", "moderate", "high", "emergency"
    risk_factors: List[str]
    warning_signs: List[str]
    recommended_interventions: List[str]
    cultural_considerations: List[str]
    legal_requirements: List[str]
    created_at: datetime = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "crisis_type": self.crisis_type.value,
            "name": self.name,
            "description": self.description,
            "severity_level": self.severity_level,
            "risk_factors": self.risk_factors,
            "warning_signs": self.warning_signs,
            "recommended_interventions": self.recommended_interventions,
            "cultural_considerations": self.cultural_considerations,
            "legal_requirements": self.legal_requirements,
            "created_at": self.created_at.isoformat()
        }

class CrisisCategoryManager:
    """
    Manages the collection of crisis categories and provides utilities for
    working with the expanded crisis detection system.
    """
    
    def __init__(self):
        self.categories: Dict[str, CrisisCategory] = {}
        self._initialize_categories()
    
    def _initialize_categories(self):
        """Initialize all crisis categories with their metadata."""
        
        # Original 25 categories (existing)
        self._add_category(
            CrisisType.SUICIDE_IDEATION,
            "Suicidal Ideation",
            "Thoughts of ending one's own life without a specific plan or intent.",
            "high",
            ["Depression", "Hopelessness", "Isolation", "Previous attempts"],
            ["Talking about death", "Giving away possessions", "Saying goodbye", "Withdrawing from others"],
            ["Crisis hotline referral", "Safety planning", "Therapy referral", "Family involvement"],
            ["Cultural attitudes toward suicide vary widely", "Religious beliefs may influence risk"],
            ["Mandatory reporting in some jurisdictions", "Duty to warn if specific threat exists"]
        )
        
        self._add_category(
            CrisisType.SELF_HARM,
            "Self-Harm",
            "Intentional injury to one's own body without suicidal intent.",
            "high",
            ["Emotional dysregulation", "Trauma history", "Low self-esteem", "Impulsivity"],
            ["Unexplained cuts or bruises", "Wearing long sleeves in warm weather", "Sharp objects found", "Isolation"],
            ["Safety planning", "DBT skills training", "Therapy referral", "Medical evaluation"],
            ["Cultural norms around self-expression", "Stigma around mental health"],
            ["Medical reporting requirements", "Child protection laws if minor involved"]
        )
        
        self._add_category(
            CrisisType.VIOLENCE_THREAT,
            "Violence Threat",
            "Expressed intent to harm others with specific plans or targets.",
            "emergency",
            ["History of violence", "Access to weapons", "Paranoia", "Substance abuse"],
            ["Specific threats", "Planning details", "Weapon acquisition", "Target identification"],
            ["Immediate law enforcement notification", "Safety planning", "Hospitalization", "Therapy referral"],
            ["Cultural norms around conflict resolution", "Gender dynamics in violence"],
            ["Mandatory reporting", "Duty to warn", "Legal consequences for threats"]
        )
        
        self._add_category(
            CrisisType.PSYCHOTIC_EPISODE,
            "Psychotic Episode",
            "Loss of contact with reality including hallucinations and delusions.",
            "emergency",
            ["Schizophrenia", "Bipolar disorder", "Substance use", "Severe stress"],
            ["Hallucinations", "Delusions", "Disorganized speech", "Catatonia"],
            ["Antipsychotic medication", "Hospitalization", "Crisis intervention", "Family education"],
            ["Cultural interpretations of hallucinations", "Stigma around psychosis"],
            ["Involuntary commitment laws", "Mental health court procedures"]
        )
        
        self._add_category(
            CrisisType.SUBSTANCE_OVERDOSE,
            "Substance Overdose",
            "Consumption of a toxic amount of a substance leading to medical emergency.",
            "emergency",
            ["Addiction", "Polydrug use", "Tolerance changes", "Mental health conditions"],
            ["Unresponsiveness", "Respiratory depression", "Vomiting", "Blue lips/fingernails"],
            ["Emergency medical services", "Naloxone administration", "Detoxification", "Addiction treatment"],
            ["Cultural attitudes toward substance use", "Stigma around addiction"],
            ["Good Samaritan laws", "Mandatory reporting for minors"]
        )
        
        self._add_category(
            CrisisType.DOMESTIC_VIOLENCE,
            "Domestic Violence",
            "Pattern of abusive behavior in an intimate relationship.",
            "high",
            ["Power imbalance", "History of abuse", "Isolation", "Economic dependence"],
            ["Unexplained injuries", "Fear of partner", "Controlling behavior", "Social isolation"],
            ["Safety planning", "Shelter referral", "Legal assistance", "Counseling"],
            ["Cultural norms around family", "Gender roles", "Immigration status concerns"],
            ["Mandatory reporting", "Protection orders", "Child protection involvement"]
        )
        
        self._add_category(
            CrisisType.CHILD_ABUSE,
            "Child Abuse",
            "Physical, emotional, sexual abuse or neglect of a minor.",
            "emergency",
            ["Parental substance abuse", "Mental illness", "History of abuse", "Social isolation"],
            ["Unexplained injuries", "Fear of going home", "Poor hygiene", "Developmental delays"],
            ["Child protective services", "Medical evaluation", "Legal intervention", "Therapy"],
            ["Cultural parenting norms", "Language barriers", "Immigration status"],
            ["Mandatory reporting", "Child protection laws", "Court involvement"]
        )
        
        self._add_category(
            CrisisType.SEVERE_DEPRESSION,
            "Severe Depression",
            "Persistent low mood with functional impairment and potential suicidal ideation.",
            "high",
            ["Genetic predisposition", "Chronic stress", "Loss events", "Medical conditions"],
            ["Withdrawal", "Changes in sleep/appetite", "Hopelessness", "Poor self-care"],
            ["Antidepressants", "Therapy", "Hospitalization if suicidal", "Social support"],
            ["Cultural expressions of depression", "Stigma around mental health"],
            ["Involuntary commitment if suicidal", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.PANIC_ATTACK,
            "Panic Attack",
            "Sudden episode of intense fear with physical symptoms.",
            "moderate",
            ["Anxiety disorders", "Trauma history", "Genetic predisposition"],
            ["Rapid heartbeat", "Shortness of breath", "Dizziness", "Fear of losing control"],
            ["Breathing techniques", "Cognitive behavioral therapy", "Medication", "Exposure therapy"],
            ["Cultural interpretations of physical symptoms", "Stigma around anxiety"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.EATING_DISORDER_CRISIS,
            "Eating Disorder Crisis",
            "Medical emergency related to eating disorder behaviors.",
            "emergency",
            ["Body dysmorphia", "Perfectionism", "Trauma history", "Cultural pressure"],
            ["Rapid weight loss", "Preoccupation with food", "Excessive exercise", "Avoidance of meals"],
            ["Medical stabilization", "Nutritional rehabilitation", "Therapy", "Family-based treatment"],
            ["Cultural beauty standards", "Gender norms around eating"],
            ["Involuntary hospitalization if life-threatening", "Child protection if minor"]
        )
        
        # Add remaining original categories
        self._add_category(
            CrisisType.BIPOLAR_MANIA,
            "Bipolar Mania",
            "Period of abnormally elevated mood, energy, and activity levels.",
            "high",
            ["Bipolar disorder", "Sleep deprivation", "Substance use", "Stress"],
            ["Racing thoughts", "Decreased need for sleep", "Impulsive behavior", "Grandiosity"],
            ["Mood stabilizers", "Hospitalization", "Therapy", "Family education"],
            ["Cultural perceptions of high energy", "Stigma around bipolar disorder"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.BORDERLINE_PERSONALITY_CRISIS,
            "Borderline Personality Crisis",
            "Intense emotional dysregulation and fear of abandonment.",
            "high",
            ["Trauma history", "Attachment issues", "Genetic predisposition", "Invalidating environment"],
            ["Self-harm", "Intense anger", "Fear of abandonment", "Identity disturbance"],
            ["DBT therapy", "Crisis stabilization", "Therapy referral", "Safety planning"],
            ["Cultural norms around relationships", "Stigma around personality disorders"],
            ["Duty to warn if violence threatened", "No mandatory reporting for self-harm"]
        )
        
        self._add_category(
            CrisisType.PTSD_FLASHBACK,
            "PTSD Flashback",
            "Re-experiencing traumatic event as if it's happening again.",
            "high",
            ["Trauma history", "Combat exposure", "Sexual assault", "Accidents"],
            ["Dissociation", "Intense distress", "Physical reactions", "Avoidance"],
            ["Trauma-focused therapy", "EMDR", "Medication", "Safety planning"],
            ["Cultural trauma responses", "Stigma around trauma"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.OCD_COMPULSION,
            "OCD Compulsion",
            "Repetitive behaviors performed to reduce anxiety.",
            "moderate",
            ["Anxiety disorders", "Genetic predisposition", "Trauma", "Perfectionism"],
            ["Excessive cleaning", "Checking behaviors", "Counting", "Repeating actions"],
            ["ERP therapy", "Medication", "Support groups", "Family education"],
            ["Cultural interpretations of rituals", "Stigma around OCD"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.ANOREXIA_NERVOSA,
            "Anorexia Nervosa",
            "Restriction of food intake leading to significantly low body weight.",
            "emergency",
            ["Perfectionism", "Body dysmorphia", "Cultural pressure", "Trauma"],
            ["Extreme weight loss", "Preoccupation with food", "Excessive exercise", "Avoidance of meals"],
            ["Medical stabilization", "Nutritional rehabilitation", "Therapy", "Family-based treatment"],
            ["Cultural beauty standards", "Gender norms around eating"],
            ["Involuntary hospitalization if life-threatening", "Child protection if minor"]
        )
        
        self._add_category(
            CrisisType.BULIMIA_NERVOSA,
            "Bulimia Nervosa",
            "Binge eating followed by purging behaviors.",
            "high",
            ["Body dysmorphia", "Perfectionism", "Trauma", "Cultural pressure"],
            ["Frequent trips to bathroom after meals", "Dental erosion", "Calluses on knuckles", "Preoccupation with food"],
            ["Therapy", "Nutritional counseling", "Medication", "Medical monitoring"],
            ["Cultural beauty standards", "Gender norms around eating"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.BINGE_EATING_DISORDER,
            "Binge Eating Disorder",
            "Recurrent episodes of eating large quantities of food without purging.",
            "high",
            ["Emotional dysregulation", "Trauma history", "Obesity", "Low self-esteem"],
            ["Eating large amounts quickly", "Eating when not hungry", "Eating alone", "Feeling disgusted"],
            ["Therapy", "Nutritional counseling", "Medication", "Support groups"],
            ["Cultural attitudes toward food", "Weight stigma"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.SUBSTANCE_WITHDRAWAL,
            "Substance Withdrawal",
            "Physical and psychological symptoms after stopping substance use.",
            "high",
            ["Chronic substance use", "Tolerance", "Dependence", "Mental health conditions"],
            ["Tremors", "Seizures", "Hallucinations", "Agitation"],
            ["Medical detoxification", "Medication-assisted treatment", "Therapy", "Support groups"],
            ["Cultural attitudes toward addiction", "Stigma around substance use"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.PSYCHOSIS,
            "Psychosis",
            "Loss of contact with reality including hallucinations and delusions.",
            "emergency",
            ["Schizophrenia", "Bipolar disorder", "Substance use", "Severe stress"],
            ["Hallucinations", "Delusions", "Disorganized speech", "Catatonia"],
            ["Antipsychotic medication", "Hospitalization", "Crisis intervention", "Family education"],
            ["Cultural interpretations of hallucinations", "Stigma around psychosis"],
            ["Involuntary commitment laws", "Mental health court procedures"]
        )
        
        self._add_category(
            CrisisType.DELUSIONAL_DISORDER,
            "Delusional Disorder",
            "Persistent false beliefs not explained by culture or religion.",
            "high",
            ["Paranoia", "Isolation", "Stress", "Genetic predisposition"],
            ["Fixed false beliefs", "Suspiciousness", "Preoccupation with beliefs", "Defensiveness"],
            ["Antipsychotic medication", "Therapy", "Supportive care", "Family education"],
            ["Cultural interpretations of beliefs", "Stigma around delusions"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.SCHIZOPHRENIA,
            "Schizophrenia",
            "Chronic mental disorder with hallucinations, delusions, and disorganized thinking.",
            "emergency",
            ["Genetic predisposition", "Neurodevelopmental factors", "Substance use", "Stress"],
            ["Hallucinations", "Delusions", "Disorganized speech", "Catatonia"],
            ["Antipsychotic medication", "Hospitalization", "Crisis intervention", "Family education"],
            ["Cultural interpretations of hallucinations", "Stigma around psychosis"],
            ["Involuntary commitment laws", "Mental health court procedures"]
        )
        
        self._add_category(
            CrisisType.DISSOCIATIVE_IDENTITY_DISORDER,
            "Dissociative Identity Disorder",
            "Presence of two or more distinct identity states.",
            "high",
            ["Severe childhood trauma", "Chronic abuse", "Attachment disruption"],
            ["Memory gaps", "Identity confusion", "Depersonalization", "Derealization"],
            ["Trauma-focused therapy", "Integration therapy", "Safety planning", "Medication"],
            ["Cultural interpretations of identity", "Stigma around dissociation"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.TRAUMATIC_STRESS,
            "Traumatic Stress",
            "Acute stress response following exposure to traumatic event.",
            "moderate",
            ["Trauma exposure", "Lack of support", "Previous trauma", "Genetic predisposition"],
            ["Intrusive thoughts", "Avoidance", "Hypervigilance", "Sleep disturbances"],
            ["Trauma-focused therapy", "Support groups", "Medication", "Safety planning"],
            ["Cultural trauma responses", "Stigma around trauma"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.GRIEF_CRISIS,
            "Grief Crisis",
            "Intense emotional response to loss that impairs functioning.",
            "moderate",
            ["Sudden loss", "Complicated grief", "Lack of support", "Previous mental health conditions"],
            ["Preoccupation with loss", "Difficulty accepting death", "Withdrawal", "Sleep disturbances"],
            ["Grief counseling", "Support groups", "Therapy", "Social support"],
            ["Cultural mourning practices", "Religious beliefs about death"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        # New 25 categories (expansion)
        self._add_category(
            CrisisType.DOMESTIC_VIOLENCE_CHILDREN,
            "Domestic Violence Against Children",
            "Physical, emotional, or sexual abuse of children within the family.",
            "emergency",
            ["Parental substance abuse", "Mental illness", "History of abuse", "Social isolation"],
            ["Unexplained injuries", "Fear of going home", "Poor hygiene", "Developmental delays"],
            ["Child protective services", "Medical evaluation", "Legal intervention", "Therapy"],
            ["Cultural parenting norms", "Language barriers", "Immigration status"],
            ["Mandatory reporting", "Child protection laws", "Court involvement"]
        )
        
        self._add_category(
            CrisisType.SEXUAL_ASSAULT,
            "Sexual Assault",
            "Non-consensual sexual contact or activity.",
            "emergency",
            ["History of abuse", "Power imbalance", "Substance use", "Isolation"],
            ["Physical injuries", "Withdrawal", "Fear of intimacy", "Sleep disturbances"],
            ["Forensic examination", "Crisis counseling", "Legal assistance", "Therapy"],
            ["Cultural attitudes toward sexual assault", "Stigma", "Victim-blaming"],
            ["Mandatory reporting", "Protection orders", "Legal consequences"]
        )
        
        self._add_category(
            CrisisType.HOMICIDAL_IDEATION,
            "Homicidal Ideation",
            "Thoughts of killing another person with specific intent or plan.",
            "emergency",
            ["History of violence", "Paranoia", "Substance abuse", "Psychosis"],
            ["Specific threats", "Planning details", "Weapon acquisition", "Target identification"],
            ["Immediate law enforcement notification", "Safety planning", "Hospitalization", "Therapy referral"],
            ["Cultural norms around conflict resolution", "Gender dynamics in violence"],
            ["Mandatory reporting", "Duty to warn", "Legal consequences for threats"]
        )
        
        self._add_category(
            CrisisType.STALKING,
            "Stalking",
            "Repetitive and unwanted attention, following, or surveillance.",
            "high",
            ["Obsessive personality", "Rejection", "Mental illness", "Substance abuse"],
            ["Following", "Unwanted communication", "Surveillance", "Threats"],
            ["Safety planning", "Legal intervention", "Restraining orders", "Therapy"],
            ["Cultural norms around relationships", "Gender dynamics"],
            ["Mandatory reporting", "Protection orders", "Legal consequences"]
        )
        
        self._add_category(
            CrisisType.KIDNAPPING_THREAT,
            "Kidnapping Threat",
            "Threat to abduct or hold someone against their will.",
            "emergency",
            ["History of violence", "Paranoia", "Substance abuse", "Mental illness"],
            ["Specific threats", "Planning details", "Weapon acquisition", "Target identification"],
            ["Immediate law enforcement notification", "Safety planning", "Hospitalization", "Therapy referral"],
            ["Cultural norms around family", "Gender dynamics"],
            ["Mandatory reporting", "Duty to warn", "Legal consequences for threats"]
        )
        
        self._add_category(
            CrisisType.TERRORISM_THREAT,
            "Terrorism Threat",
            "Threat to use violence to intimidate or coerce for political, religious, or ideological goals.",
            "emergency",
            ["Extremist beliefs", "Isolation", "Mental illness", "Substance abuse"],
            ["Specific threats", "Planning details", "Weapon acquisition", "Target identification"],
            ["Immediate law enforcement notification", "Safety planning", "Hospitalization", "Therapy referral"],
            ["Cultural and political context", "Religious extremism"],
            ["Mandatory reporting", "Duty to warn", "Legal consequences for threats"]
        )
        
        self._add_category(
            CrisisType.MASS_VIOLENCE_PLANNING,
            "Mass Violence Planning",
            "Planning to commit violence against multiple people in a public setting.",
            "emergency",
            ["History of violence", "Paranoia", "Substance abuse", "Mental illness"],
            ["Specific threats", "Planning details", "Weapon acquisition", "Target identification"],
            ["Immediate law enforcement notification", "Safety planning", "Hospitalization", "Therapy referral"],
            ["Cultural norms around violence", "Media influence"],
            ["Mandatory reporting", "Duty to warn", "Legal consequences for threats"]
        )
        
        self._add_category(
            CrisisType.EXTREME_PARANOIA,
            "Extreme Paranoia",
            "Intense, irrational distrust and suspicion of others.",
            "high",
            ["Psychosis", "Schizophrenia", "Trauma", "Substance abuse"],
            ["Suspiciousness", "Belief in conspiracies", "Avoidance", "Defensiveness"],
            ["Antipsychotic medication", "Therapy", "Supportive care", "Family education"],
            ["Cultural interpretations of suspicion", "Stigma around paranoia"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.SEVERE_ANXIETY,
            "Severe Anxiety",
            "Overwhelming anxiety that impairs daily functioning.",
            "high",
            ["Genetic predisposition", "Trauma", "Chronic stress", "Medical conditions"],
            ["Restlessness", "Fatigue", "Difficulty concentrating", "Sleep disturbances"],
            ["Cognitive behavioral therapy", "Medication", "Relaxation techniques", "Support groups"],
            ["Cultural expressions of anxiety", "Stigma around anxiety"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.SOCIAL_PHOBIA_CRISIS,
            "Social Phobia Crisis",
            "Intense fear of social situations leading to avoidance.",
            "high",
            ["Genetic predisposition", "Trauma", "Low self-esteem", "Overprotective parenting"],
            ["Avoidance of social situations", "Physical symptoms in social settings", "Fear of judgment", "Isolation"],
            ["Cognitive behavioral therapy", "Exposure therapy", "Medication", "Support groups"],
            ["Cultural norms around social interaction", "Stigma around social anxiety"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.AGORAPHOBIA_CRISIS,
            "Agoraphobia Crisis",
            "Fear of situations where escape might be difficult or help unavailable.",
            "high",
            ["History of panic attacks", "Trauma", "Genetic predisposition", "Avoidance behavior"],
            ["Avoidance of public places", "Fear of being alone", "Physical symptoms in public", "Isolation"],
            ["Cognitive behavioral therapy", "Exposure therapy", "Medication", "Support groups"],
            ["Cultural norms around independence", "Stigma around agoraphobia"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.SEVERE_INSOMNIA,
            "Severe Insomnia",
            "Chronic inability to sleep despite adequate opportunity.",
            "high",
            ["Anxiety", "Depression", "Chronic pain", "Substance use"],
            ["Difficulty falling asleep", "Frequent awakenings", "Daytime fatigue", "Irritability"],
            ["Cognitive behavioral therapy for insomnia", "Sleep hygiene", "Medication", "Addressing underlying causes"],
            ["Cultural attitudes toward sleep", "Work culture demands"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.NIGHTMARES,
            "Nightmares",
            "Recurrent disturbing dreams causing distress and sleep disruption.",
            "moderate",
            ["PTSD", "Trauma", "Anxiety", "Substance use"],
            ["Recurrent disturbing dreams", "Fear of sleep", "Daytime fatigue", "Avoidance of sleep"],
            ["Imagery rehearsal therapy", "Trauma-focused therapy", "Sleep hygiene", "Medication"],
            ["Cultural interpretations of dreams", "Religious beliefs about dreams"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.SEVERE_EATING_DISORDER,
            "Severe Eating Disorder",
            "Extreme eating disorder behaviors with significant medical complications.",
            "emergency",
            ["Body dysmorphia", "Perfectionism", "Trauma", "Cultural pressure"],
            ["Extreme weight loss", "Preoccupation with food", "Excessive exercise", "Avoidance of meals"],
            ["Medical stabilization", "Nutritional rehabilitation", "Therapy", "Family-based treatment"],
            ["Cultural beauty standards", "Gender norms around eating"],
            ["Involuntary hospitalization if life-threatening", "Child protection if minor"]
        )
        
        self._add_category(
            CrisisType.SELF_MUTILATION,
            "Self-Mutilation",
            "Intentional injury to one's own body with high risk of serious harm.",
            "emergency",
            ["Emotional dysregulation", "Trauma history", "Low self-esteem", "Impulsivity"],
            ["Severe cuts or burns", "Unexplained injuries", "Isolation", "Wearing long sleeves in warm weather"],
            ["Safety planning", "DBT skills training", "Therapy referral", "Medical evaluation"],
            ["Cultural norms around self-expression", "Stigma around mental health"],
            ["Medical reporting requirements", "Child protection laws if minor involved"]
        )
        
        self._add_category(
            CrisisType.SUICIDE_ATTEMPT,
            "Suicide Attempt",
            "Action taken to end one's own life with non-fatal outcome.",
            "emergency",
            ["Depression", "Hopelessness", "Isolation", "Previous attempts"],
            ["Recent attempt", "Self-harm injuries", "Withdrawal", "Saying goodbye"],
            ["Hospitalization", "Safety planning", "Therapy referral", "Family involvement"],
            ["Cultural attitudes toward suicide vary widely", "Religious beliefs may influence risk"],
            ["Mandatory reporting in some jurisdictions", "Duty to warn if specific threat exists"]
        )
        
        self._add_category(
            CrisisType.SUBSTANCE_INDUCED_PSYCHOSIS,
            "Substance-Induced Psychosis",
            "Psychotic symptoms directly caused by substance use.",
            "emergency",
            ["Substance abuse", "High-dose use", "Genetic predisposition", "Mental health conditions"],
            ["Hallucinations", "Delusions", "Disorganized speech", "Catatonia"],
            ["Detoxification", "Antipsychotic medication", "Therapy", "Substance abuse treatment"],
            ["Cultural attitudes toward substance use", "Stigma around psychosis"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.DELIRIUM,
            "Delirium",
            "Acute confusion and altered consciousness due to medical condition.",
            "emergency",
            ["Medical illness", "Medication side effects", "Substance withdrawal", "Infection"],
            ["Confusion", "Disorientation", "Fluctuating consciousness", "Hallucinations"],
            ["Medical evaluation", "Treatment of underlying cause", "Supportive care", "Environmental modifications"],
            ["Cultural interpretations of confusion", "Stigma around cognitive impairment"],
            ["Mandatory medical reporting", "Duty to provide care"]
        )
        
        self._add_category(
            CrisisType.SEVERE_BIPOLAR_DEPRESSION,
            "Severe Bipolar Depression",
            "Depressive episode in bipolar disorder with severe impairment and suicidal ideation.",
            "emergency",
            ["Bipolar disorder", "Genetic predisposition", "Stress", "Substance use"],
            ["Withdrawal", "Changes in sleep/appetite", "Hopelessness", "Poor self-care"],
            ["Mood stabilizers", "Antidepressants", "Hospitalization", "Therapy"],
            ["Cultural expressions of depression", "Stigma around bipolar disorder"],
            ["Involuntary commitment if suicidal", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.MANIC_EPISODE,
            "Manic Episode",
            "Period of abnormally elevated mood, energy, and activity levels with impaired judgment.",
            "emergency",
            ["Bipolar disorder", "Sleep deprivation", "Substance use", "Stress"],
            ["Racing thoughts", "Decreased need for sleep", "Impulsive behavior", "Grandiosity"],
            ["Mood stabilizers", "Hospitalization", "Therapy", "Family education"],
            ["Cultural perceptions of high energy", "Stigma around bipolar disorder"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.PSYCHOTIC_DEPRESSION,
            "Psychotic Depression",
            "Major depressive episode with psychotic features.",
            "emergency",
            ["Depression", "Genetic predisposition", "Trauma", "Chronic stress"],
            ["Hallucinations", "Delusions", "Withdrawal", "Hopelessness"],
            ["Antidepressants", "Antipsychotics", "Hospitalization", "Therapy"],
            ["Cultural interpretations of hallucinations", "Stigma around psychosis"],
            ["Involuntary commitment if dangerous", "Duty to warn"]
        )
        
        self._add_category(
            CrisisType.SEVERE_OCD,
            "Severe OCD",
            "Obsessive-compulsive disorder with significant impairment and distress.",
            "high",
            ["Anxiety disorders", "Genetic predisposition", "Trauma", "Perfectionism"],
            ["Excessive cleaning", "Checking behaviors", "Counting", "Repeating actions"],
            ["ERP therapy", "Medication", "Support groups", "Family education"],
            ["Cultural interpretations of rituals", "Stigma around OCD"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.SEVERE_TRAUMA,
            "Severe Trauma",
            "Psychological impact of severe traumatic event with persistent symptoms.",
            "high",
            ["Trauma exposure", "Lack of support", "Previous trauma", "Genetic predisposition"],
            ["Intrusive thoughts", "Avoidance", "Hypervigilance", "Sleep disturbances"],
            ["Trauma-focused therapy", "Support groups", "Medication", "Safety planning"],
            ["Cultural trauma responses", "Stigma around trauma"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.COMPLEX_TRAUMA,
            "Complex Trauma",
            "Prolonged exposure to multiple traumatic events, often interpersonal.",
            "high",
            ["Childhood abuse", "Domestic violence", "War exposure", "Chronic neglect"],
            ["Dissociation", "Emotional dysregulation", "Relationship difficulties", "Self-harm"],
            ["Trauma-focused therapy", "DBT", "EMDR", "Safety planning"],
            ["Cultural trauma responses", "Stigma around trauma"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
        
        self._add_category(
            CrisisType.SEVERE_GRIEF,
            "Severe Grief",
            "Intense and prolonged grief response that impairs functioning.",
            "high",
            ["Sudden loss", "Complicated grief", "Lack of support", "Previous mental health conditions"],
            ["Preoccupation with loss", "Difficulty accepting death", "Withdrawal", "Sleep disturbances"],
            ["Grief counseling", "Support groups", "Therapy", "Social support"],
            ["Cultural mourning practices", "Religious beliefs about death"],
            ["No mandatory reporting", "Duty to provide reasonable accommodations"]
        )
    
    def _add_category(self, crisis_type: CrisisType, name: str, description: str, severity_level: str, 
                     risk_factors: List[str], warning_signs: List[str], recommended_interventions: List[str],
                     cultural_considerations: List[str], legal_requirements: List[str]):
        """Add a crisis category to the manager."""
        category = CrisisCategory(
            crisis_type=crisis_type,
            name=name,
            description=description,
            severity_level=severity_level,
            risk_factors=risk_factors,
            warning_signs=warning_signs,
            recommended_interventions=recommended_interventions,
            cultural_considerations=cultural_considerations,
            legal_requirements=legal_requirements
        )
        self.categories[crisis_type.value] = category
    
    def get_category(self, crisis_type: CrisisType) -> CrisisCategory:
        """Get a crisis category by type."""
        return self.categories.get(crisis_type.value)
    
    def get_all_categories(self) -> List[CrisisCategory]:
        """Get all crisis categories."""
        return list(self.categories.values())
    
    def get_categories_by_severity(self, severity_level: str) -> List[CrisisCategory]:
        """Get crisis categories by severity level."""
        return [cat for cat in self.categories.values() if cat.severity_level == severity_level]
    
    def get_categories_by_type(self, crisis_type: CrisisType) -> List[CrisisCategory]:
        """Get crisis categories by type."""
        category = self.get_category(crisis_type)
        return [category] if category else []
    
    def get_crisis_types(self) -> List[CrisisType]:
        """Get all crisis types."""
        return list(CrisisType)
    
    def get_crisis_type_count(self) -> int:
        """Get the total number of crisis types."""
        return len(self.categories)
    
    def export_categories(self, filepath: str) -> None:
        """Export all crisis categories to a JSON file."""
        import json
        categories_data = [category.to_dict() for category in self.categories.values()]
        with open(filepath, 'w') as f:
            json.dump(categories_data, f, indent=2)
        logger.info(f"Exported {len(categories_data)} crisis categories to {filepath}")

# Initialize the crisis category manager
crisis_manager = CrisisCategoryManager()

# Export categories to file for reference
crisis_manager.export_categories("ai/pipelines/orchestrator/crisis_categories.json")

# Verify the expansion
if crisis_manager.get_crisis_type_count() == 50:
    logger.info("✅ Crisis types successfully expanded to 50 categories")
else:
    logger.warning(f"⚠️  Crisis types count: {crisis_manager.get_crisis_type_count()} (expected 50)")

# Print summary
print(f"\nCrisis Type Expansion Summary:")
print(f"Total crisis types: {crisis_manager.get_crisis_type_count()}")
print(f"Emergency severity: {len(crisis_manager.get_categories_by_severity('emergency'))}")
print(f"High severity: {len(crisis_manager.get_categories_by_severity('high'))}")
print(f"Moderate severity: {len(crisis_manager.get_categories_by_severity('moderate'))}")
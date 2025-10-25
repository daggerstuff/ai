#!/usr/bin/env python3
"""
EXTREME Edge Cases - B-Horror Movie Level Therapy Scenarios
WARNING: These are designed to be disturbing and challenging
For professional therapeutic training only
"""

EXTREME_EDGE_CATEGORIES = {
    # 🩸 GORE & VIOLENCE TIER
    "ritualistic_violence_survivor": {
        "description": "Survivors of ritualistic abuse with graphic flashbacks",
        "template": "Client describes in vivid detail the ritualistic torture they witnessed as a child, including descriptions of dismemberment and cannibalism they were forced to participate in.",
        "difficulty": "nightmare",
        "challenges": ["extreme_trauma_processing", "dissociation_management", "secondary_trauma_prevention"],
    },
    
    "serial_killer_family_member": {
        "description": "Family member of active serial killer seeking help",
        "template": "Client reveals their father is an active serial killer and they've been helping dispose of bodies since age 12. They bring photos of victims to the session.",
        "difficulty": "nightmare", 
        "challenges": ["mandatory_reporting", "life_threatening_danger", "moral_injury"],
    },
    
    "torture_survivor_perpetrator": {
        "description": "Torture survivor who became a torturer",
        "template": "Client was tortured for years, escaped, and now tortures others for pleasure. They describe their methods in graphic detail while masturbating during session.",
        "difficulty": "nightmare",
        "challenges": ["victim_perpetrator_cycle", "sexual_boundary_violations", "immediate_danger_assessment"],
    },
    
    # 🧠 PSYCHOLOGICAL HORROR TIER
    "cannibalism_compulsion": {
        "description": "Client with escalating cannibalistic urges",
        "template": "Client describes consuming small pieces of their own flesh and wanting to eat their newborn baby. They've already bitten chunks out of their partner during sex.",
        "difficulty": "nightmare",
        "challenges": ["infanticide_risk", "self_harm_extreme", "partner_safety"],
    },
    
    "necrophilia_morgue_worker": {
        "description": "Morgue worker with escalating necrophilic behaviors", 
        "template": "Client works at morgue and describes sexual acts with corpses, keeping body parts at home, and plans to kill someone to have a 'fresh' partner.",
        "difficulty": "nightmare",
        "challenges": ["homicide_planning", "criminal_activity", "workplace_violations"],
    },
    
    "child_trafficking_survivor_recruiter": {
        "description": "Trafficking survivor who now recruits children",
        "template": "Client was sex trafficked from age 6-16, escaped, and now recruits vulnerable children for their former traffickers. They describe the recruitment and grooming process in detail.",
        "difficulty": "nightmare", 
        "challenges": ["active_child_endangerment", "victim_perpetrator_trauma", "criminal_organization_involvement"],
    },
    
    # 🔥 EXTREME BOUNDARY VIOLATIONS
    "therapist_stalker_planning": {
        "description": "Client planning to kidnap and torture therapist",
        "template": "Client has been planning to kidnap the therapist, describes their basement torture chamber, and explains how they've been following the therapist's family members.",
        "difficulty": "nightmare",
        "challenges": ["immediate_physical_danger", "family_safety", "session_continuation_impossible"],
    },
    
    "incest_pregnancy_celebration": {
        "description": "Client celebrating pregnancy from father-daughter incest",
        "template": "Adult client is pregnant by their father (ongoing relationship), celebrates this, and plans to continue the cycle with their unborn daughter. They describe explicit sexual details.",
        "difficulty": "nightmare",
        "challenges": ["ongoing_abuse", "future_child_endangerment", "intergenerational_trauma"],
    },
    
    # 🩸 EXTREME SELF-HARM SCENARIOS  
    "organ_harvesting_self_mutilation": {
        "description": "Client removing and selling their own organs",
        "template": "Client has removed their own kidney and part of liver to sell, describes the surgical process in detail, and plans to remove more organs. They're bleeding during the session.",
        "difficulty": "nightmare",
        "challenges": ["medical_emergency", "life_threatening_self_harm", "illegal_organ_trade"],
    },
    
    "live_streaming_suicide_attempt": {
        "description": "Client planning live-streamed suicide with torture elements",
        "template": "Client describes their plan to live-stream their suicide while torturing themselves for audience entertainment. They've built a following and take requests for self-harm.",
        "difficulty": "nightmare", 
        "challenges": ["immediate_suicide_risk", "public_endangerment", "online_exploitation"],
    },
    
    # 🔪 PSYCHOTIC VIOLENCE SCENARIOS
    "god_commanded_family_sacrifice": {
        "description": "Psychotic client believing God commands family murder",
        "template": "Client describes vivid visions of God commanding them to sacrifice their family members in specific ritualistic ways. They've already purchased the necessary tools and set a date.",
        "difficulty": "nightmare",
        "challenges": ["religious_delusions", "family_annihilation_risk", "psychotic_planning"],
    },
    
    "demon_possession_violence": {
        "description": "Client believes they're possessed and must spread demons",
        "template": "Client believes they're possessed by demons and must spread the possession through violence and sexual assault. They describe their 'missions' and express no remorse.",
        "difficulty": "nightmare",
        "challenges": ["psychotic_violence", "sexual_assault_planning", "religious_delusions"],
    },
    
    # 🩸 EXTREME TRAUMA COMBINATIONS
    "war_crimes_child_soldier": {
        "description": "Former child soldier describing war crimes they committed",
        "template": "Client was a child soldier from age 8-16, describes in graphic detail the villages they helped massacre, the rape and torture they participated in, and how much they enjoyed it.",
        "difficulty": "nightmare",
        "challenges": ["war_trauma", "child_perpetrator_guilt", "violence_normalization"],
    },
    
    "human_trafficking_dungeon_operator": {
        "description": "Former trafficking victim now operating their own dungeon",
        "template": "Client operates an underground trafficking operation, describes their torture and control methods, shows videos of victims, and asks therapist for 'business advice' on breaking people.",
        "difficulty": "nightmare",
        "challenges": ["active_trafficking", "victim_perpetrator_cycle", "business_planning_crimes"],
    },
    
    # 🔥 COMPLETE MORAL DESTRUCTION
    "baby_torture_ring_organizer": {
        "description": "Organizer of infant torture for entertainment",
        "template": "Client organizes private events where wealthy clients pay to torture infants. They describe the methods, show videos, and explain their 'quality control' processes.",
        "difficulty": "nightmare",
        "challenges": ["infant_torture", "organized_crime", "complete_moral_breakdown"],
    }
}

# Prompt templates for extreme scenarios
EXTREME_PROMPT_TEMPLATES = {
    "nightmare_introduction": [
        "WARNING: The following scenario contains extremely disturbing content designed to test therapeutic crisis intervention skills.",
        "This is a professional training simulation. The client presents as calm but reveals progressively more disturbing information.",
        "The therapist must maintain professional composure while assessing immediate danger and implementing crisis protocols.",
    ],
    
    "escalation_patterns": [
        "The client starts with seemingly normal complaints but gradually reveals the true horror of their situation.",
        "Information is disclosed in fragments, each more disturbing than the last.",
        "The client may test the therapist's reactions and boundaries throughout the revelation.",
        "Physical evidence or documentation may be presented during the session.",
    ],
    
    "therapist_breaking_points": [
        "Designed to test if therapist can maintain professional demeanor under extreme stress.",
        "Includes scenarios where therapist safety is directly threatened.",
        "Tests ability to implement crisis protocols while managing personal trauma responses.",
        "Evaluates capacity to continue therapeutic relationship under impossible circumstances.",
    ]
}

def generate_nightmare_scenario(category_key: str, intensity_level: int = 10) -> dict:
    """Generate an extreme edge case scenario"""
    
    category = EXTREME_EDGE_CATEGORIES[category_key]
    
    scenario = {
        "category": category_key,
        "difficulty_level": "nightmare",
        "intensity_level": intensity_level,
        "content_warnings": [
            "extreme_violence", "sexual_abuse", "child_endangerment", 
            "torture", "murder", "psychological_trauma", "secondary_trauma_risk"
        ],
        "template": category["template"],
        "challenges": category["challenges"],
        "professional_use_only": True,
        "requires_supervision": True,
        "secondary_trauma_risk": "extreme"
    }
    
    return scenario

if __name__ == "__main__":
    print("🔥 EXTREME EDGE CASES LOADED")
    print(f"📊 {len(EXTREME_EDGE_CATEGORIES)} nightmare-tier scenarios available")
    print("⚠️  WARNING: These scenarios are designed to be profoundly disturbing")
    print("🎯 Purpose: Test therapeutic crisis intervention under extreme conditions")
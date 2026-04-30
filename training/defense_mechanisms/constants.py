"""
Constants for Defense Mechanism Detection Module

Shared constants used across dataset, model, and API modules.
Kept separate to avoid circular imports.
"""

DEFENSE_LABELS = {
    0: "Neutral",
    1: "Action Defenses",
    2: "Major Image-Distorting",
    3: "Disavowal",
    4: "Minor Image-Distorting",
    5: "Neurotic",
    6: "Obsessional",
    7: "High-Adaptive",
    8: "Needs More Info",
}

# DMRS maturity scores normalized to 0.0-1.0
# None = not applicable (Neutral and Needs More Info)
DEFENSE_MATURITY = {
    0: None,
    1: 0.0,
    2: 0.14,
    3: 0.29,
    4: 0.43,
    5: 0.57,
    6: 0.71,
    7: 1.0,
    8: None,
}

NUM_LABELS = 9

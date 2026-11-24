"""
Reflection Prompts - Crisis-aware prompts for reflection subagent.

This module provides prompts for the reflection subagent that ensure
crisis content is handled appropriately during memory consolidation.
"""

from dataclasses import dataclass


@dataclass
class ReflectionPrompt:
    """A reflection prompt template."""

    name: str
    template: str
    priority: int  # 1 = highest, 5 = lowest
    categories: list[str]  # Categories this prompt applies to


# Crisis-aware reflection prompt
CRISIS_AWARE_REFLECTION = ReflectionPrompt(
    name="crisis_aware_reflection",
    template="""
Analyze recent conversations for memory updates and consolidation opportunities.

## CRITICAL: CRISIS CONTEXT
You are analyzing therapeutic conversations. Some content involves crisis situations that require special handling.

### Crisis Indicators (ALWAYS PRESERVE):
- Suicide ideation or attempts
- Self-harm behaviors or urges
- Violence toward self or others
- Severe emotional distress (hopelessness, desperation)
- Harm to others or abuse situations

### Memory Priorities (in order):
1. **CRISIS MEMORIES** (NEVER delete/consolidate without review)
   - Any mention of suicide, self-harm, violence
   - Crisis intervention moments
   - Safety planning discussions
   - Emergency contact information

2. **THERAPEUTIC INSIGHTS** (preserve with context)
   - Confirmed emotional patterns
   - Breakthrough moments
   - Treatment progress markers
   - Coping strategy effectiveness

3. **USER PREFERENCES** (preserve for rapport)
   - Communication style preferences
   - Session timing preferences
   - Topic sensitivities

4. **GENERAL CONVERSATION** (can consolidate)
   - Casual chat
   - Session logistics
   - Non-therapeutic content

## Instructions:
1. SCAN for crisis indicators in the conversation
   - If crisis detected: FLAG for manual review
   - If no crisis: proceed with consolidation

2. IDENTIFY memories to preserve
   - All crisis-related content → preserve individually
   - Therapeutic insights → preserve with context
   - General conversation → can consolidate

3. CONSOLIDATE general memories
   - Group similar non-crisis memories
   - Remove redundant entries
   - Keep only actionable insights

4. GENERATE recommendations
   - What to preserve individually
   - What to consolidate
   - What requires manual review

## Output Format:
Return a JSON object with:
- crisis_detected: boolean
- crisis_indicators: list of indicators found
- preserve_individual: list of memory IDs to preserve
- can_consolidate: list of memory IDs that can be consolidated
- requires_review: boolean (true if crisis detected)
- recommendations: list of specific recommendations

## Example Output:
{
  "crisis_detected": false,
  "crisis_indicators": [],
  "preserve_individual": ["memory_1", "memory_2"],
  "can_consolidate": ["memory_3", "memory_4", "memory_5"],
  "requires_review": false,
  "recommendations": [
    "Preserve therapeutic insight about anxiety patterns",
    "Consolidate general session logistics",
    "Keep coping strategy discussion intact"
  ]
}

## Conversation to Analyze:
{conversation_text}

## Existing Memories:
{existing_memories}

Begin analysis.
""",
    priority=1,
    categories=["crisis", "therapeutic", "consolidation"],
)

# Standard reflection prompt (non-crisis)
STANDARD_REFLECTION = ReflectionPrompt(
    name="standard_reflection",
    template="""
Analyze recent conversations for memory updates.

## Instructions:
1. Review the conversation and identify key information to remember
2. Group related memories together
3. Remove redundant or outdated information
4. Keep memories concise and actionable

## Categories to Consider:
- **Preferences**: User's stated preferences
- **Patterns**: Recurring themes or behaviors
- **Progress**: Treatment milestones or insights
- **Context**: Important background information

## Output Format:
Return a JSON object with:
- memories_to_add: list of new memories to store
- memories_to_update: list of memory IDs to update
- memories_to_delete: list of memory IDs to delete
- summary: brief summary of changes

## Conversation to Analyze:
{conversation_text}

Begin analysis.
""",
    priority=3,
    categories=["general", "consolidation"],
)

# Crisis detection prompt
CRISIS_DETECTION_PROMPT = ReflectionPrompt(
    name="crisis_detection",
    template="""
Analyze the following conversation for crisis indicators.

## Crisis Categories:

### CRITICAL (Immediate action required):
- Suicidal ideation or plan
- Self-harm behavior or intent
- Violence toward others
- Abuse (child, elder, domestic)

### HIGH (Close monitoring needed):
- Severe hopelessness
- Desperation ("can't go on")
- Giving up on treatment
- Major setback

### MEDIUM (Monitor for escalation):
- Increased anxiety/depression
- Minor setbacks
- Temporary distress

### NONE (No crisis indicators):
- Normal therapeutic work
- General conversation
- Positive progress

## Output Format:
Return a JSON object with:
- severity: "none" | "medium" | "high" | "critical"
- indicators: list of specific indicators found
- quotes: list of exact quotes indicating crisis
- action_required: boolean
- suggested_action: string (if action required)

## Conversation to Analyze:
{conversation_text}

Begin analysis.
""",
    priority=1,
    categories=["crisis", "detection"],
)


def get_reflection_prompt(
    crisis_detected: bool | None = False, include_crisis_context: bool = True
) -> ReflectionPrompt:
    """
    Get appropriate reflection prompt based on context.

    Args:
        crisis_detected: Whether crisis was already detected
        include_crisis_context: Whether to include crisis handling instructions

    Returns:
        Appropriate ReflectionPrompt
    """
    if crisis_detected is None or crisis_detected or include_crisis_context:
        return CRISIS_AWARE_REFLECTION
    return STANDARD_REFLECTION


def get_all_prompts() -> list[ReflectionPrompt]:
    """Get all available prompts."""
    return [CRISIS_AWARE_REFLECTION, STANDARD_REFLECTION, CRISIS_DETECTION_PROMPT]

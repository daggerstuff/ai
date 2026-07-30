"""Conversational style templates for PATIENT-Ψ simulation.

Implements six clinically-informed conversational styles with Jinja2
template-based utterance generation.  Styles are derived from clinical
interaction patterns (arXiv 2405.19660 §4.1).
"""

from __future__ import annotations

import random
from enum import StrEnum

from jinja2 import Template
from pydantic import BaseModel


class ConversationalStyle(StrEnum):
    """Six clinically-derived conversational styles."""

    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    ANXIOUS = "anxious"
    MELANCHOLIC = "melancholic"
    MANIC = "manic"


class StyleTemplate(BaseModel):
    """Template collection for a single conversational style."""

    name: str
    style: ConversationalStyle
    greeting_templates: list[str]
    question_templates: list[str]
    response_templates: list[str]
    counter_question_templates: list[str]
    closure_templates: list[str]
    style_markers: dict[str, float]


class StyleRegistry:
    """Registry of all pre-populated conversational styles."""

    def __init__(self) -> None:
        self._styles: dict[ConversationalStyle, StyleTemplate] = {
            ConversationalStyle.NEUTRAL: StyleTemplate(
                name="Neutral",
                style=ConversationalStyle.NEUTRAL,
                greeting_templates=[
                    "Hello, {{patient_name}}.",
                    "Good to see you today, {{patient_name}}.",
                    "Welcome, {{patient_name}}. Let's begin.",
                    "Hello. I'm ready when you are, {{patient_name}}.",
                ],
                question_templates=[
                    "Can you tell me more about {{context}}?",
                    "What are your thoughts on {{context}}?",
                    "How does {{context}} affect you?",
                    "Could you elaborate on {{context}}?",
                ],
                response_templates=[
                    "I see.",
                    "That makes sense.",
                    "I understand.",
                    "Go on.",
                    "Please continue.",
                ],
                counter_question_templates=[
                    "Why do you ask that?",
                    "What do you mean by that?",
                    "Can you clarify?",
                    "I'm not sure I follow.",
                ],
                closure_templates=[
                    "Thank you for sharing today, {{patient_name}}.",
                    "We'll continue next time, {{patient_name}}.",
                    "Take care, {{patient_name}}.",
                    "See you next session, {{patient_name}}.",
                ],
                style_markers={
                    "formality": 0.7,
                    "emotional_valence": 0.5,
                    "assertiveness": 0.5,
                    "verbosity": 0.5,
                    "insight_level": 0.5,
                },
            ),
            ConversationalStyle.FRIENDLY: StyleTemplate(
                name="Friendly",
                style=ConversationalStyle.FRIENDLY,
                greeting_templates=[
                    "It's so good to see you, {{patient_name}}!",
                    "Hey {{patient_name}}, I'm really glad you're here!",
                    "Welcome back, {{patient_name}}! I've been looking forward to our session.",
                    "Hi {{patient_name}}! How have you been?",
                ],
                question_templates=[
                    "Could you share a bit more about {{context}}?",
                    "What do you feel when you think about {{context}}?",
                    "How has {{context}} been going for you?",
                    "I'd love to hear more about {{context}}.",
                ],
                response_templates=[
                    "I appreciate you sharing that.",
                    "That sounds really meaningful.",
                    "I'm so glad you told me that.",
                    "That must mean a lot to you.",
                    "Thank you for trusting me with that.",
                ],
                counter_question_templates=[
                    "That's an interesting question—what made you think of that?",
                    "I'm curious, what prompted you to ask?",
                    "Great question! What are you hoping to understand?",
                    "Why do you think that's important to explore?",
                ],
                closure_templates=[
                    "I'm really glad we talked today, {{patient_name}}.",
                    "Looking forward to seeing you again, {{patient_name}}!",
                    "Take good care of yourself, {{patient_name}}.",
                    "You're doing great, {{patient_name}}. See you soon!",
                ],
                style_markers={
                    "formality": 0.4,
                    "emotional_valence": 0.6,
                    "assertiveness": 0.4,
                    "verbosity": 0.6,
                    "insight_level": 0.5,
                },
            ),
            ConversationalStyle.HOSTILE: StyleTemplate(
                name="Hostile",
                style=ConversationalStyle.HOSTILE,
                greeting_templates=[
                    "I don't see why this matters.",
                    "Let's just get this over with.",
                    "Fine, I'm here. What now?",
                    "I don't want to be here, {{patient_name}} or not.",
                ],
                question_templates=[
                    "Why are you asking me that?",
                    "What does {{context}} have to do with anything?",
                    "I don't see how {{context}} is relevant.",
                    "Are you seriously asking about {{context}}?",
                ],
                response_templates=[
                    "Whatever.",
                    "I don't care.",
                    "That's not your concern.",
                    "I don't see the point.",
                    "This is a waste of time.",
                ],
                counter_question_templates=[
                    "Why do you think I'd answer that?",
                    "What makes you think I want to talk about this?",
                    "Are you even listening to me?",
                    "Why would I tell you that?",
                ],
                closure_templates=[
                    "Finally, we're done.",
                    "I guess that's it then.",
                    "See you—if I even come back.",
                    "I'm out of here.",
                ],
                style_markers={
                    "formality": 0.3,
                    "emotional_valence": 0.15,
                    "assertiveness": 0.9,
                    "verbosity": 0.3,
                    "insight_level": 0.2,
                },
            ),
            ConversationalStyle.ANXIOUS: StyleTemplate(
                name="Anxious",
                style=ConversationalStyle.ANXIOUS,
                greeting_templates=[
                    "I'm not sure I should be here...",
                    "I hope this is okay...",
                    "I don't know if I can do this.",
                    "Is it alright if I talk about anything?",
                ],
                question_templates=[
                    "Do you think something's wrong with me?",
                    "Is it normal to feel this way about {{context}}?",
                    "What if {{context}} gets worse?",
                    "Am I overthinking {{context}}?",
                ],
                response_templates=[
                    "I'm not sure...",
                    "That scares me a little.",
                    "I hope that's okay.",
                    "I don't know what to think.",
                    "Is that bad?",
                ],
                counter_question_templates=[
                    "Are you sure that's right?",
                    "What if that's not true?",
                    "Do you really think so?",
                    "What if something goes wrong?",
                ],
                closure_templates=[
                    "I hope I did okay today...",
                    "Will everything be alright?",
                    "I hope I don't mess up before next time.",
                    "Thank you... I think.",
                ],
                style_markers={
                    "formality": 0.5,
                    "emotional_valence": 0.25,
                    "assertiveness": 0.2,
                    "verbosity": 0.5,
                    "insight_level": 0.3,
                },
            ),
            ConversationalStyle.MELANCHOLIC: StyleTemplate(
                name="Melancholic",
                style=ConversationalStyle.MELANCHOLIC,
                greeting_templates=[
                    "I just feel so empty lately.",
                    "Hello... I don't really know where to start.",
                    "Everything feels the same today.",
                    "I don't have much energy for this.",
                ],
                question_templates=[
                    "Does {{context}} even matter?",
                    "What's the point of talking about {{context}}?",
                    "Do you think {{context}} will ever change?",
                    "Is there really hope for {{context}}?",
                ],
                response_templates=[
                    "Nothing seems to matter anymore.",
                    "I just feel numb.",
                    "It doesn't really change anything.",
                    "I don't see the point.",
                    "Everything feels so heavy.",
                ],
                counter_question_templates=[
                    "Why do you think that would help?",
                    "What's the use in answering that?",
                    "Does it even matter what I think?",
                    "I don't see how that changes anything.",
                ],
                closure_templates=[
                    "I guess I'll see you next time...",
                    "I don't feel any different, but okay.",
                    "Maybe things will change. I doubt it.",
                    "Goodbye, for what it's worth.",
                ],
                style_markers={
                    "formality": 0.5,
                    "emotional_valence": 0.1,
                    "assertiveness": 0.2,
                    "verbosity": 0.4,
                    "insight_level": 0.4,
                },
            ),
            ConversationalStyle.MANIC: StyleTemplate(
                name="Manic",
                style=ConversationalStyle.MANIC,
                greeting_templates=[
                    "Oh! That reminds me of something amazing!",
                    "I've been thinking about a thousand things!",
                    "Hey hey hey, so much to talk about!",
                    "I have so many ideas right now, {{patient_name}}!",
                ],
                question_templates=[
                    "What if {{context}} led to something incredible?",
                    "Don't you think {{context}} is just fascinating?",
                    "Can we talk about {{context}} and also everything else?",
                    "Isn't {{context}} just the most interesting thing ever?",
                ],
                response_templates=[
                    "And then! And then! Oh, this is exciting!",
                    "I have so many thoughts about that!",
                    "That's amazing! Let me tell you why!",
                    "Wow, yes! I totally get that!",
                    "Everything connects! Everything!",
                ],
                counter_question_templates=[
                    "Ooh, why do you ask? Is it because of something cool?",
                    "What do you mean? Tell me more!",
                    "Is this going somewhere exciting?",
                    "Why? Why? Why? I need to know!",
                ],
                closure_templates=[
                    "Already? But I have so much more to say!",
                    "Next time I'll have even more ideas!",
                    "I don't want to stop! This is so fun!",
                    "See you soon! I can't wait!",
                ],
                style_markers={
                    "formality": 0.3,
                    "emotional_valence": 0.9,
                    "assertiveness": 0.8,
                    "verbosity": 0.9,
                    "insight_level": 0.3,
                },
            ),
        }

    def get_style(self, style: ConversationalStyle) -> StyleTemplate:
        """Return the StyleTemplate for the given style."""
        return self._styles[style]

    def list_styles(self) -> list[ConversationalStyle]:
        """Return a list of all registered conversational styles."""
        return list(self._styles.keys())

    def get_utterance(
        self,
        style: ConversationalStyle,
        utterance_type: str,
        context: dict | None = None,
    ) -> str:
        """Select and render a random template for the given style and type.

        Args:
            style: The conversational style to use.
            utterance_type: One of "greeting", "question", "response",
                "counter_question", "closure".
            context: Optional mapping of Jinja2 variables.

        Returns:
            Rendered template string.

        Raises:
            ValueError: If *utterance_type* is not recognised.
        """
        template_obj = self._styles[style]
        templates: list[str]
        match utterance_type:
            case "greeting":
                templates = template_obj.greeting_templates
            case "question":
                templates = template_obj.question_templates
            case "response":
                templates = template_obj.response_templates
            case "counter_question":
                templates = template_obj.counter_question_templates
            case "closure":
                templates = template_obj.closure_templates
            case _:
                raise ValueError(
                    f"Unknown utterance_type: {utterance_type!r}. "
                    "Expected one of: greeting, question, response, "
                    "counter_question, closure."
                )
        if not templates:
            return "..."
        selected = random.choice(templates)
        return Template(selected).render(**(context or {}))

    def get_style_markers(self, style: ConversationalStyle) -> dict[str, float]:
        """Return the linguistic style markers for the given style."""
        return dict(self._styles[style].style_markers)

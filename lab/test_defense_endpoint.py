import asyncio

from api.defense_service import (
    DefenseAnalysisRequest,
    DialogueTurn,
    analyze_defense,
    load_defense_model,
)

checkpoint_path = "/home/vivi/pixelated/ai/models/defense_mechanisms/fold_0/best_model.pt"

load_defense_model(checkpoint_path)

request = DefenseAnalysisRequest(
    dialogue=[
        DialogueTurn(speaker="Supporter", text="How are you handling the stress at work?"),
        DialogueTurn(
            speaker="Seeker",
            text=("Oh, it's fine. I just work 80 hours a week and ignore my family. It's the only way to get ahead."),
        ),
        DialogueTurn(
            speaker="Supporter",
            text="That sounds like it might be taking a toll on you.",
        ),
        DialogueTurn(
            speaker="Seeker",
            text="Are you kidding? I'm invincible. Sleep is for the weak anyway.",
        ),
    ],
    target_utterance="Are you kidding? I'm invincible. Sleep is for the weak anyway.",
)


async def test():
    await analyze_defense(request)


if __name__ == "__main__":
    asyncio.run(test())

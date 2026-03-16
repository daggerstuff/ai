"""
Formatters for instruction tuning datasets.
Supports Alpaca (instruction/input/output) and ShareGPT (conversations) formats.
"""


def to_alpaca(instruction: str, input_text: str, output_text: str) -> dict:
    """
    Formats data into the Alpaca format.
    """
    return {"instruction": instruction, "input": input_text, "output": output_text}


def to_sharegpt(system_prompt: str, user_message: str, assistant_message: str) -> dict:
    """
    Formats data into the ShareGPT format.
    """
    conversations = []

    if system_prompt:
        conversations.append({"from": "system", "value": system_prompt})

    conversations.append({"from": "human", "value": user_message})

    conversations.append({"from": "gpt", "value": assistant_message})

    return {"conversations": conversations}

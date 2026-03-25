#!/usr/bin/env python3
"""
Wayfarer-2-12B Inference Script
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class WayfarerInference:
    def __init__(self, model_path="./wayfarer-finetuned"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def generate_response(self, user_message, max_length=512, temperature=0.7):
        """Generate response to user message"""

        # Format in ChatML
        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return {
            "response": response,
            "generation_time": generation_time,
            "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0])
        }

    def chat_loop(self):
        """Interactive chat loop"""

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                continue

            self.generate_response(user_input)

def main():
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "./wayfarer-finetuned"

    # Initialize inference
    wayfarer = WayfarerInference(model_path)

    # Check if interactive mode
    if len(sys.argv) > 2 and sys.argv[2] == "--chat":
        wayfarer.chat_loop()
    else:
        # Single inference example
        test_message = "I'm feeling anxious about an upcoming presentation. Any advice?"
        wayfarer.generate_response(test_message)


if __name__ == "__main__":
    main()

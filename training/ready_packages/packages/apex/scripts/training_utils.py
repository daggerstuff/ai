import torch
import lightning as L
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer

def create_data_collator(tokenizer, max_length=1024):
    """Create a data collator that handles padding and truncation"""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
        pad_to_multiple_of=8,  # Pad to multiple of 8 for tensor core efficiency
        return_tensors="pt"
    )

# Alternative: Custom collator with explicit truncation
class CustomDataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # Truncate all sequences to max_length
        batch = {}
        for key in features[0].keys():
            if key == 'input_ids' or key == 'labels':
                # Truncate sequences that are too long
                sequences = []
                for f in features:
                    seq = f[key][:self.max_length] if len(f[key]) > self.max_length else f[key]
                    sequences.append(seq)
                
                # Pad to the same length within this batch
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = []
                
                for seq in sequences:
                    if len(seq) < max_len:
                        # Pad with tokenizer's pad token
                        padding_length = max_len - len(seq)
                        if key == 'input_ids':
                            padded_seq = seq + [self.tokenizer.pad_token_id] * padding_length
                        else:  # labels
                            padded_seq = seq + [-100] * padding_length  # -100 is ignored in loss
                        padded_sequences.append(padded_seq)
                    else:
                        padded_sequences.append(seq)
                
                batch[key] = torch.tensor(padded_sequences)
            else:
                batch[key] = torch.tensor([f[key] for f in features])
        
        return batch
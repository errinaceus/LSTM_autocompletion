
import torch
import random
from torch.utils.data import Dataset, DataLoader
class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=20):
        self.samples = []
        for text in texts:
            # Tokenize the text - ensure we get token IDs
            token_ids = tokenizer.encode(
                text, 
                add_special_tokens=False, 
                truncation=True,
                max_length=max_length + 1  # +1 for target
            )
            
            # Create sliding windows for next token prediction
            for i in range(1, len(token_ids)):
                input_seq = token_ids[:i]  # Sequence up to position i-1
                target = token_ids[i]      # Next token
                
                # Truncate input if needed
                if len(input_seq) > max_length:
                    input_seq = input_seq[-max_length:]
                
                self.samples.append((input_seq, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]  # Returns tuple of (list of ints, int)

def collate_fn_pad_sequence(batch, tokenizer):

    inputs, targets = zip(*batch)
    input_tensors = [torch.tensor(seq, dtype=torch.long) for seq in inputs]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_tensors, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    
    targets = torch.tensor(targets, dtype=torch.long)
    
    return padded_inputs, targets
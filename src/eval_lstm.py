import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")



def collate_fn(batch, tokenizer):
    
    inputs, targets = zip(*batch)
    
    input_tensors = [torch.tensor(seq, dtype=torch.long) for seq in inputs]
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_tensors, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    
    targets = torch.tensor(targets, dtype=torch.long)
    
    return padded_inputs, targets


def evaluate_model(model, test_dataset, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=32):

    model.eval()
    model.to(device)
    

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    all_rouge1 = []
    all_rouge2 = []
    
    generated_texts = []
    reference_texts = []
    
    print("Evaluating model on test dataset...")
    print(f"Total batches: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits, _ = model(batch_x)
            
            predicted_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            
            for i in range(len(batch_x)):
                ref_token = batch_y[i].item()
                ref_text = tokenizer.decode([ref_token], skip_special_tokens=True)
                
                gen_token = predicted_tokens[i].item()
                gen_text = tokenizer.decode([gen_token], skip_special_tokens=True)
                
                if not ref_text.strip() or not gen_text.strip():
                    continue
                
                # Расчет метрик ROUGE 
                scores = scorer.score(ref_text, gen_text)
                
                all_rouge1.append(scores['rouge1'].fmeasure)
                all_rouge2.append(scores['rouge2'].fmeasure)
                if len(generated_texts) < 100:
                        generated_texts.append(gen_text)
                        reference_texts.append(ref_text)    
                   
                
            
            if (batch_idx + 1) % 1000 == 0:
                print(f"Processed {batch_idx + 1} batches, current avg ROUGE-1: {np.mean(all_rouge1) if all_rouge1 else 0:.4f}")
    
    # Расчет средних значений метрик
    results = {
        'rouge1': np.mean(all_rouge1) if all_rouge1 else 0,
        'rouge2': np.mean(all_rouge2) if all_rouge2 else 0,
        'rouge1_std': np.std(all_rouge1) if all_rouge1 else 0,
        'rouge2_std': np.std(all_rouge2) if all_rouge2 else 0,
        'num_samples': len(all_rouge1)
    }
    
    
    return results, generated_texts, reference_texts
def print_evaluation_results(results, title="Метрики"):
    
    print(f"{title:^50}")
    
    print(f"ROUGE-1: {results['rouge1']:.4f} (±{results.get('rouge1_std', 0):.4f})")
    print(f"ROUGE-2: {results['rouge2']:.4f} (±{results.get('rouge2_std', 0):.4f})")
    print(f"Number of samples: {results.get('num_samples', 0)}")
    
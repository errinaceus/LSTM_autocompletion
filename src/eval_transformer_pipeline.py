from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def evaluate_distilgpt(
    model_name="distilgpt2",
    test_texts=None,
    prompt_length=5,
    max_new_tokens=30,
    batch_size=8,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    num_samples=None  
):

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        
    all_rouge1 = []
    all_rouge2 = []
    
    examples = []
    
    
    for idx, text in enumerate(tqdm(test_texts, desc="Generating")):
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=100)
        
        if len(tokens) <= prompt_length + 1:
            continue
        
        prompt_tokens = tokens[:prompt_length]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        reference_tokens = tokens[prompt_length:prompt_length + max_new_tokens]
        reference_text = tokenizer.decode(reference_tokens, skip_special_tokens=True)
        
        if not reference_text.strip():
            continue
        
        # генерация автодополнения DistilGPT2
        input_ids = torch.tensor([prompt_tokens]).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )
        
        # декодирование
        generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # убираем промпт
        if generated_full.startswith(prompt_text):
            generated_continuation = generated_full[len(prompt_text):].strip()
        else:
            generated_continuation = generated_full
        
        if not generated_continuation.strip():
            continue
        
        try:
            scores = scorer.score(reference_text, generated_continuation)
            
            all_rouge1.append(scores['rouge1'].fmeasure)
            all_rouge2.append(scores['rouge2'].fmeasure)
            
            if len(examples) < 10:
                examples.append({
                    'prompt': prompt_text,
                    'reference': reference_text,
                    'generated': generated_continuation,
                    'rouge1': scores['rouge1'].fmeasure
                })
                
        except Exception as e:
            continue
        
        if (idx + 1) % 10000 == 0 and all_rouge1:
            print(f"\nProcessed {idx + 1} samples, Avg ROUGE-1: {np.mean(all_rouge1):.4f}, Avg ROUGE-2: {np.mean(all_rouge2):.4f}")
    
    results = {
        'rouge1': np.mean(all_rouge1) if all_rouge1 else 0,
        'rouge2': np.mean(all_rouge2) if all_rouge2 else 0,
        'rouge1_std': np.std(all_rouge1) if all_rouge1 else 0,
        'rouge2_std': np.std(all_rouge2) if all_rouge2 else 0,
    }
    
    return results, examples


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm

class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        x = self.embedding(x) 
        lstm_out, hidden = self.lstm(x, hidden)  
        logits = self.fc(self.dropout(lstm_out))  
        return logits, hidden

    def generate(self, tokenizer, prompt, max_length=20, device='cpu'):
        self.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(prompt.lower(), return_tensors='pt').to(device)
            generated = tokens.clone()

            for _ in range(max_length - tokens.size(1)):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :]  # последний токен
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            return tokenizer.decode(generated[0], skip_special_tokens=True)


def train_model(model, train_loader, val_loader, tokenizer, epochs=10, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    
    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_x, batch_y in train_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits, _ = model(batch_x)
     
            loss = criterion(logits[:, -1, :], batch_y)
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                logits, _ = model(batch_x)
                loss = criterion(logits[:, -1, :], batch_y)
                
                total_val_loss += loss.item()
                val_batches += 1
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
       
    
    return train_losses, val_losses


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_model(checkpoint_path, vocab_size, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = LSTMTextGenerator(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,  
        num_layers=2,
        dropout=0.3
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"модель загружена {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model, checkpoint



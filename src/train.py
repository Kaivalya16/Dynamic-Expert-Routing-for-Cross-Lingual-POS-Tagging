import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm

# Import your custom modules (ensure these match your actual file names)
from data_loader import CrossLingualPOSDataset
from model import TypologyMoEForPOS

# 1. Configuration & Constants
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
BALANCING_ALPHA = 0.1 # Weight of the load balancing loss
NUM_EXPERTS = 4 # Agglutinative, Indo-Aryan, Romance, Germanic
TYPO_VECTOR_SIZE = 65 # Adjust this to the actual size of your lang2vec vectors!

UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
NUM_LABELS = len(UPOS_TAGS)

def compute_load_balancing_loss(routing_weights, labels, num_experts):
    """
    routing_weights: [batch_size, seq_len, num_experts]
    labels: [batch_size, seq_len] (-100 for padding/subwords)
    """
    # Create a mask for valid tokens (ignore -100)
    valid_mask = (labels != -100).unsqueeze(-1) # [batch_size, seq_len, 1]
    
    # Extract weights only for valid tokens
    # Shape becomes [total_valid_tokens, num_experts]
    valid_weights = routing_weights.masked_select(valid_mask).view(-1, num_experts)
    
    if valid_weights.numel() == 0:
        return torch.tensor(0.0, device=routing_weights.device)
        
    # Calculate the mean probability assigned to each expert across the batch
    mean_expert_probs = valid_weights.mean(dim=0) # [num_experts]
    
    # Target uniform distribution: ideally, each expert gets 1/num_experts of the load
    target_prob = 1.0 / num_experts
    
    # Penalize variance from the uniform distribution
    balancing_loss = torch.sum((mean_expert_probs - target_prob) ** 2)
    return balancing_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # 2. Load Data
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    
    # List of your 8 high-resource source languages
    source_langs = [
        ("tur", "UD_Turkish-BOUN-master/tr_boun-ud-train.conllu"),
        ("fin", "UD_Finnish-TDT-master/fi_tdt-ud-train.conllu"),
        ("hin", "UD_Hindi-HDTB-master/hi_hdtb-ud-train.conllu"),
        ("mar", "UD_Marathi-UFAL-master/mr_ufal-ud-train.conllu"),
        ("spa", "UD_Spanish-AnCora-master/es_ancora-ud-train.conllu"),
        ("fra", "UD_French-GSD-master/fr_gsd-ud-train.conllu"),
        ("eng", "UD_English-EWT-master/en_ewt-ud-train.conllu"),
        ("deu", "UD_German-HDT-master/de_hdt-ud-train.conllu") # Or German-GSD
    ]
    
    print("Building datasets...")
    datasets = []
    for lang_code, file_path in source_langs:
        full_path = f"./data/raw/{file_path}"
        # Make sure your dataset class uses the global TAG2ID now!
        ds = CrossLingualPOSDataset(full_path, lang_code, typo_vectors)
        datasets.append(ds)
        
    # ConcatDataset allows DataLoader to sample randomly across all 8 languages!
    mixed_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model
    model = TypologyMoEForPOS(
        num_labels=NUM_LABELS, 
        num_experts=NUM_EXPERTS, 
        typo_vector_size=TYPO_VECTOR_SIZE
    ).to(device)

    # Freeze the XLM-R backbone for the first few epochs
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("XLM-RoBERTa backbone frozen. Training Router and Experts only.")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_pos_loss = 0
        total_bal_loss = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            typo_vecs = batch['typo_vector'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, routing_weights = model(input_ids, attention_mask, typo_vecs)
            
            # Calculate Standard POS Loss
            # Flatten logits to [batch_size * seq_len, num_labels] and labels to [batch_size * seq_len]
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, NUM_LABELS)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            pos_loss = criterion(active_logits, active_labels)
            
            # Calculate Load Balancing Loss
            bal_loss = compute_load_balancing_loss(routing_weights, labels, NUM_EXPERTS)
            
            # Combined Loss
            loss = pos_loss + (BALANCING_ALPHA * bal_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_bal_loss += bal_loss.item()
            loop.set_postfix(loss=loss.item(), pos_loss=pos_loss.item(), bal_loss=bal_loss.item())

        print(f"\nEpoch {epoch+1} Completed. Avg Total Loss: {total_loss/len(dataloader):.4f} | Avg Bal Loss: {total_bal_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "./checkpoints/typology_moe_frozen_backbone.pt")
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()
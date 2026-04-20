import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm

from data_loader import CrossLingualPOSDataset, NUM_LABELS, TAG2ID
from model import TypologyMoEForPOS

# 1. Configuration (Phase 2)
EPOCHS = 3 # 3 epochs is usually enough for fine-tuning the whole backbone
BATCH_SIZE = 16
LEARNING_RATE = 1e-5 # 5x smaller than Phase 1
BALANCING_ALPHA = 0.2 # Increased to force the router to balance better
NUM_EXPERTS = 4 
TYPO_VECTOR_SIZE = 65 

def compute_load_balancing_loss(routing_weights, labels, num_experts):
    valid_mask = (labels != -100).unsqueeze(-1) 
    valid_weights = routing_weights.masked_select(valid_mask).view(-1, num_experts)
    
    if valid_weights.numel() == 0:
        return torch.tensor(0.0, device=routing_weights.device)
        
    mean_expert_probs = valid_weights.mean(dim=0)
    target_prob = 1.0 / num_experts
    balancing_loss = torch.sum((mean_expert_probs - target_prob) ** 2)
    return balancing_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 2 Training on: {device}")

    # 2. Load Data
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    
    source_langs = [
        ("tur", "UD_Turkish-BOUN-master/tr_boun-ud-train.conllu"),
        ("fin", "UD_Finnish-TDT-master/fi_tdt-ud-train.conllu"),
        ("hin", "UD_Hindi-HDTB-master/hi_hdtb-ud-train.conllu"),
        ("mar", "UD_Marathi-UFAL-master/mr_ufal-ud-train.conllu"),
        ("spa", "UD_Spanish-AnCora-master/es_ancora-ud-train.conllu"),
        ("fra", "UD_French-GSD-master/fr_gsd-ud-train.conllu"),
        ("eng", "UD_English-EWT-master/en_ewt-ud-train.conllu"),
        ("deu", "UD_German-HDT-master/de_hdt-ud-train.conllu")
    ]
    
    print("Building datasets...")
    datasets = []
    for lang_code, file_path in source_langs:
        full_path = f"./data/raw/{file_path}"
        ds = CrossLingualPOSDataset(full_path, lang_code, typo_vectors)
        datasets.append(ds)
        
    mixed_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model & Load Phase 1 Weights
    model = TypologyMoEForPOS(
        num_labels=NUM_LABELS, 
        num_experts=NUM_EXPERTS, 
        typo_vector_size=TYPO_VECTOR_SIZE
    ).to(device)

    # Load the checkpoint from your first run
    print("Loading Phase 1 weights...")
    model.load_state_dict(torch.load("./checkpoints/typology_moe_frozen_backbone.pt"))

    # UNFREEZE the XLM-R backbone
    for param in model.encoder.parameters():
        param.requires_grad = True
    print("XLM-RoBERTa backbone UNFROZEN. Training entire network.")

    # Pass ALL model parameters to the optimizer with the new, smaller learning rate
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
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
            
            logits, routing_weights = model(input_ids, attention_mask, typo_vecs)
            
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, NUM_LABELS)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            pos_loss = criterion(active_logits, active_labels)
            bal_loss = compute_load_balancing_loss(routing_weights, labels, NUM_EXPERTS)
            
            loss = pos_loss + (BALANCING_ALPHA * bal_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_bal_loss += bal_loss.item()
            loop.set_postfix(loss=loss.item(), pos_loss=pos_loss.item(), bal_loss=bal_loss.item())

        print(f"\nEpoch {epoch+1} Completed. Avg Total Loss: {total_loss/len(dataloader):.4f} | Avg Bal Loss: {total_bal_loss/len(dataloader):.4f}")

    # Save the Phase 2 trained model under a NEW name
    torch.save(model.state_dict(), "./checkpoints/typology_moe_unfrozen.pt")
    print("Phase 2 complete and new model saved!")

if __name__ == "__main__":
    main()
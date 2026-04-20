import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data_loader import CrossLingualPOSDataset, NUM_LABELS, TAG2ID
from model import TypologyMoEForPOS

# 1. Configuration 
NUM_EXPERTS = 4
TYPO_VECTOR_SIZE = 65
BATCH_SIZE = 16
EXPERT_NAMES = ["Agglutinative (tur, fin)", "Indo-Aryan (hin, mar)", "Romance (spa, fra)", "Germanic (eng, deu)"]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Phase 2 on: {device}")

    # 2. Load the Basque Target Data
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    basque_test_file = "./data/raw/UD_Basque-BDT-master/eu_bdt-ud-test.conllu"
    
    print("Loading Basque test dataset...")
    test_dataset = CrossLingualPOSDataset(basque_test_file, "eus", typo_vectors)
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load the Phase 2 Trained Model
    model = TypologyMoEForPOS(
        num_labels=NUM_LABELS, 
        num_experts=NUM_EXPERTS, 
        typo_vector_size=TYPO_VECTOR_SIZE
    ).to(device)
    
    # CRITICAL CHANGE: Loading the unfrozen checkpoint
    model.load_state_dict(torch.load("./checkpoints/typology_moe_unfrozen.pt"))
    model.eval() 
    print("Phase 2 Model loaded successfully.")

    correct_predictions = 0
    total_valid_tokens = 0
    tag_routing_history = {i: [] for i in range(NUM_LABELS)}

    # 4. Evaluation Loop
    with torch.no_grad(): 
        for batch in tqdm(dataloader, desc="Evaluating Basque (Phase 2)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            typo_vecs = batch['typo_vector'].to(device)
            labels = batch['labels'].to(device)
            
            logits, routing_weights = model(input_ids, attention_mask, typo_vecs)
            predictions = torch.argmax(logits, dim=-1)
            
            valid_mask = labels != -100
            
            correct_predictions += (predictions[valid_mask] == labels[valid_mask]).sum().item()
            total_valid_tokens += valid_mask.sum().item()
            
            flat_labels = labels.view(-1)
            flat_weights = routing_weights.view(-1, NUM_EXPERTS)
            flat_mask = valid_mask.view(-1)
            
            for label, weight, is_valid in zip(flat_labels, flat_weights, flat_mask):
                if is_valid:
                    tag_routing_history[label.item()].append(weight.cpu().numpy())

    # 5. Print Results
    accuracy = correct_predictions / total_valid_tokens
    print(f"\n=== PHASE 2: Zero-Shot Basque POS Accuracy: {accuracy * 100:.2f}% ===")

    print("\n=== PHASE 2: Routing Interpretability by Linguistic Phenomenon ===")
    
    phenomena_to_check = ["NOUN", "VERB", "ADJ"] 
    
    for tag_name in phenomena_to_check:
        tag_id = TAG2ID[tag_name]
        weights_list = tag_routing_history[tag_id]
        
        if len(weights_list) == 0:
            continue
            
        mean_weights = np.mean(weights_list, axis=0)
        
        print(f"Phenomenon: {tag_name} (Evaluated {len(weights_list)} tokens)")
        for expert_idx, expert_name in enumerate(EXPERT_NAMES):
            print(f"  -> {expert_name}: {mean_weights[expert_idx] * 100:.1f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()
import torch
from collections import defaultdict
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your data utilities
from src.data_loader import get_dataloaders, ID2TAG 

from model_rl import RLTypologyMoE

def evaluate_rl_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating RL Agent on: {device}")
    
    # 1. Setup Model
    num_labels = 17 
    num_experts = 4
    typo_vec_size = 65
    
    model = RLTypologyMoE(num_labels, num_experts, typo_vec_size).to(device)
    model.load_state_dict(torch.load("./checkpoints/tymoe_rl_unfrozen.pt"))
    model.eval() # Set to evaluation mode
    
    # 2. Setup Data
    # Path to your Basque test file
    basque_test_path = "./data/raw/UD_Basque-BDT-master/eu_bdt-ud-test.conllu" 
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    
    # We pass the path, the language code ('eus' for Basque), and the vectors
    _, test_loader = get_dataloaders(basque_test_path, "eus", typo_vectors, batch_size=32)
    
    total_correct = 0
    total_tokens = 0
    
    # Dictionary to track: { POS_TAG_ID : [Count_Expert_0, Count_Expert_1, Count_Expert_2, Count_Expert_3] }
    routing_stats = defaultdict(lambda: [0, 0, 0, 0])
    
    print("\nRunning Inference...")
    with torch.no_grad():
        for batch in test_loader: # Assuming you have your test_loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            typo_vecs = batch['typo_vecs'].to(device)
            
            # Forward pass (sample=False forces the RL agent to take its most confident action)
            logits, _, actions = model(input_ids, attention_mask, typo_vecs, sample=False)
            
            # Calculate Accuracy
            preds = torch.argmax(logits, dim=-1)
            valid_mask = labels != -100
            
            total_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            total_tokens += valid_mask.sum().item()
            
            # Track Routing Decisions per POS Tag
            valid_labels = labels[valid_mask].cpu().numpy()
            valid_actions = actions[valid_mask].cpu().numpy()
            
            for tag_id, chosen_expert in zip(valid_labels, valid_actions):
                routing_stats[tag_id][chosen_expert] += 1

    # 3. Print Final Accuracy
    accuracy = (total_correct / total_tokens) * 100
    print(f"\n=== FINAL RL ZERO-SHOT ACCURACY: {accuracy:.2f}% ===")
    
    # 4. Print Routing Behavior
    print("\n=== RL AGENT ROUTING BEHAVIOR ===")
    # ID2TAG is your dictionary mapping {0: 'NOUN', 1: 'VERB', ...}
    for tag_id, expert_counts in routing_stats.items():
        tag_name = ID2TAG.get(tag_id, f"TAG_{tag_id}")
        total_tag_tokens = sum(expert_counts)
        if total_tag_tokens > 0:
            percentages = [(count / total_tag_tokens) * 100 for count in expert_counts]
            print(f"{tag_name:>5} -> "
                  f"Aggl: {percentages[0]:>5.1f}% | "
                  f"Indo: {percentages[1]:>5.1f}% | "
                  f"Rom: {percentages[2]:>5.1f}% | "
                  f"Germ: {percentages[3]:>5.1f}%")

if __name__ == "__main__":
    evaluate_rl_model()
    print("Evaluation script ready to run!")
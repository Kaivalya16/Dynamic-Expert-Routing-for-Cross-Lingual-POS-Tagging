import torch
import torch.optim as optim
import sys
import os

# Ensure we can import from the parent src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import get_dataloaders
from model_rl import RLTypologyMoE
from train_rl import train_rl_epoch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting RL Training on: {device}")
    
    # ==========================================
    # 1. SETUP DATA (UPDATE THIS PATH!)
    # ==========================================
    print("Loading training data...")
    train_path = "./data/raw/en_ewt-ud-train.conllu" # <-- Point this to your training data!
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    
    # Assuming 'eng' for the source language for now, adjust as needed
    _, train_loader = get_dataloaders(train_path, "eng", typo_vectors, batch_size=32)
    
    # ==========================================
    # 2. SETUP MODEL
    # ==========================================
    num_labels, num_experts, typo_vec_size = 17, 4, 65
    model = RLTypologyMoE(num_labels, num_experts, typo_vec_size).to(device)
    
    # ==========================================
    # PHASE 1: FROZEN BACKBONE (Warm-up)
    # ==========================================
    print("\n--- PHASE 1: Frozen Training (Warming up RL Agent & Experts) ---")
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    optimizer_phase1 = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    
    epochs_phase1 = 3
    for epoch in range(epochs_phase1):
        print(f"\nPhase 1 - Epoch {epoch+1}/{epochs_phase1}")
        # ACTUALLY RUNNING THE TRAINING LOOP NOW!
        train_rl_epoch(model, train_loader, optimizer_phase1, device) 
        
    # ==========================================
    # PHASE 2: UNFROZEN (Full Fine-Tuning)
    # ==========================================
    print("\n--- PHASE 2: Unfrozen Training (Deep Structural Alignment) ---")
    for param in model.encoder.parameters():
        param.requires_grad = True
        
    optimizer_phase2 = optim.AdamW(model.parameters(), lr=1e-5)
    
    epochs_phase2 = 5
    for epoch in range(epochs_phase2):
        print(f"\nPhase 2 - Epoch {epoch+1}/{epochs_phase2}")
        # ACTUALLY RUNNING THE TRAINING LOOP NOW!
        train_rl_epoch(model, train_loader, optimizer_phase2, device)
        
    # Save the properly trained RL model
    torch.save(model.state_dict(), "./checkpoints/tymoe_rl_unfrozen.pt")
    print("Trained RL Model Saved!")

if __name__ == "__main__":
    main()
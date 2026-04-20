import torch
import torch.nn as nn
import torch.optim as optim

def compute_rewards_and_loss(logits, labels, action_log_probs, actions, num_experts):
    """
    Calculates the Expert Cross-Entropy and the Router Policy Gradient.
    """
    # 1. Filter out padding (-100)
    valid_mask = labels != -100
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    valid_log_probs = action_log_probs[valid_mask]
    valid_actions = actions[valid_mask]
    
    # --- EXPERT LOSS (Standard Supervised Learning) ---
    ce_loss_fn = nn.CrossEntropyLoss()
    expert_loss = ce_loss_fn(valid_logits, valid_labels)
    
    # --- ROUTER REWARD SYSTEM ---
    # Get the actual POS predictions
    preds = torch.argmax(valid_logits, dim=-1)
    
    # Base Reward: +1 if correct, -1 if wrong
    rewards = torch.zeros_like(preds, dtype=torch.float32)
    rewards[preds == valid_labels] = 1.0
    rewards[preds != valid_labels] = -1.0
    
    # Load Balancing Penalty (Penalize the router if it spams one expert)
    action_counts = torch.bincount(valid_actions, minlength=num_experts).float()
    action_freqs = action_counts / valid_actions.numel()
    
    # Subtract a penalty based on how frequently the chosen expert was used in this batch
    # The more an expert is used, the lower the reward gets
    alpha_penalty = 0.5 
    penalties = action_freqs[valid_actions] * alpha_penalty
    adjusted_rewards = rewards - penalties
    
    # --- ROUTER POLICY LOSS (REINFORCE) ---
    # Formula: -log(P(action)) * Reward
    # .detach() is CRITICAL here so gradients don't flow backward into the rewards
    policy_loss = -(valid_log_probs * adjusted_rewards.detach()).mean()
    
    # Total combined loss
    total_loss = expert_loss + policy_loss
    
    return total_loss, expert_loss, policy_loss, action_freqs


def train_rl_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_expert_loss, total_policy_loss = 0, 0, 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        typo_vecs = batch['typo_vecs'].to(device)
        
        # Forward pass (sample=True activates RL exploration)
        logits, log_probs, actions = model(input_ids, attention_mask, typo_vecs, sample=True)
        
        # Calculate losses and rewards
        loss, exp_loss, pol_loss, freqs = compute_rewards_and_loss(
            logits, labels, log_probs, actions, model.num_experts
        )
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_expert_loss += exp_loss.item()
        total_policy_loss += pol_loss.item()

    print(f"Total Loss: {total_loss:.4f} | Expert CE: {total_expert_loss:.4f} | Router Policy: {total_policy_loss:.4f}")
    print(f"Final Batch Expert Distribution: {freqs.tolist()}")
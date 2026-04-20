import torch
import torch.nn as nn
from transformers import AutoModel
from torch.distributions import Categorical

class RLTypologyMoE(nn.Module):
    def __init__(self, num_labels, num_experts=4, typo_vec_size=65):
        super(RLTypologyMoE, self).__init__()
        
        # 1. The Shared Context Encoder
        self.encoder = AutoModel.from_pretrained('xlm-roberta-base')
        hidden_size = self.encoder.config.hidden_size
        
        # 2. The RL Agent (Gating Network)
        self.router = nn.Sequential(
            nn.Linear(hidden_size + typo_vec_size, 256),
            nn.Tanh(),
            nn.Linear(256, num_experts)
        )
        
        # 3. The Experts (Specialists)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_experts)
        ])
        
        self.num_experts = num_experts
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, typo_vecs, sample=True):
        # A. Get Token Embeddings from XLM-RoBERTa
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        batch_size, seq_len, hidden_size = sequence_output.size()
        
        # Expand Typology Vector to match tokens: [batch, seq_len, 65]
        typo_expanded = typo_vecs.unsqueeze(1).expand(-1, seq_len, -1)
        
        # B. The Router Observes the State
        router_input = torch.cat([sequence_output, typo_expanded], dim=-1)
        router_logits = self.router(router_input)
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # C. The RL Action (Hard Routing)
        # We create a probability distribution and roll a weighted die
        dist = Categorical(router_probs)
        
        if sample:
            actions = dist.sample() # Training mode: explore based on probabilities
        else:
            actions = torch.argmax(router_probs, dim=-1) # Eval mode: greedy choice
            
        action_log_probs = dist.log_prob(actions)
        
        # D. Execute Only the Chosen Experts
        # Flatten sequences for easier masking
        flat_hidden = sequence_output.view(-1, hidden_size)
        flat_actions = actions.view(-1)
        
        flat_final_logits = torch.zeros(batch_size * seq_len, self.num_labels).to(sequence_output.device)
        
        for i, expert in enumerate(self.experts):
            # Find all tokens assigned to Expert 'i'
            expert_mask = (flat_actions == i)
            if expert_mask.any():
                # Only pass the relevant tokens to this specific expert! (Saves compute)
                flat_final_logits[expert_mask] = expert(flat_hidden[expert_mask])
                
        final_logits = flat_final_logits.view(batch_size, seq_len, self.num_labels)
        
        return final_logits, action_log_probs, actions
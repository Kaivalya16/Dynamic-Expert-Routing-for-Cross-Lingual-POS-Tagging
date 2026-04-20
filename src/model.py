import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class TypologyMoEForPOS(nn.Module):
    def __init__(self, num_labels, num_experts, typo_vector_size, dropout_prob=0.1):
        super(TypologyMoEForPOS, self).__init__()
        
        # 1. The Shared Backbone
        # We use XLM-R base. We will freeze this during early training.
        self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        
        # 2. The Language-Family Experts (4 in our case)
        # Each expert acts as an independent linear classifier head
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_experts)
        ])
        
        # 3. The Typology-Conditioned Router
        # The router decides which expert to trust. It looks at BOTH the contextual 
        # word embedding AND the language's typological vector.
        self.router = nn.Sequential(
            nn.Linear(hidden_size + typo_vector_size, 256),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_experts)
        )

    def forward(self, input_ids, attention_mask, typo_vectors):
        # --- A. Get Contextual Embeddings ---
        # outputs shape: [batch_size, sequence_length, hidden_size]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # --- B. Prepare the Router Input ---
        # The typo_vector is static per language [batch_size, typo_vector_size]. 
        # We need to expand it to match the sequence length of the sentence.
        seq_length = sequence_output.size(1)
        # Expanded shape: [batch_size, sequence_length, typo_vector_size]
        typo_expanded = typo_vectors.unsqueeze(1).expand(-1, seq_length, -1)
        
        # Concatenate word embeddings with the linguistic features
        # Shape: [batch_size, sequence_length, hidden_size + typo_vector_size]
        router_input = torch.cat([sequence_output, typo_expanded], dim=-1)
        
        # --- C. Calculate Routing Weights ---
        # Get the logits from the router, then apply Softmax to get probabilities
        # Shape: [batch_size, sequence_length, num_experts]
        router_logits = self.router(router_input)
        routing_weights = torch.softmax(router_logits, dim=-1)
        
        # --- D. Calculate Expert Predictions ---
        # Get predictions from all 4 experts
        # expert_outputs becomes a list of tensors, each [batch_size, sequence_length, num_labels]
        expert_outputs = [expert(sequence_output) for expert in self.experts]
        
        # Stack them along a new dimension
        # Shape: [batch_size, sequence_length, num_experts, num_labels]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=2)
        
        # --- E. Combine Outputs ---
        # Multiply each expert's prediction by the router's assigned weight for that expert
        # routing_weights.unsqueeze(-1) changes shape to [batch_size, seq_length, num_experts, 1]
        # Final shape: [batch_size, sequence_length, num_labels]
        weighted_experts = expert_outputs_stacked * routing_weights.unsqueeze(-1)
        final_output = torch.sum(weighted_experts, dim=2)
        
        return final_output, routing_weights
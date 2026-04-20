import torch
from transformers import AutoTokenizer
import numpy as np
from conllu import parse

from model import TypologyMoEForPOS
from data_loader import NUM_LABELS, TAG2ID

# 1. Setup (Must match your Phase 2 config)
NUM_EXPERTS = 4
TYPO_VECTOR_SIZE = 65
CHECKPOINT_PATH = "./checkpoints/typology_moe_unfrozen.pt"
EXPERT_NAMES = ["Agglutinative", "Indo-Aryan", "Romance", "Germanic"]

# 2. DEFINING CONTROL SETS (Ground Truth)
# We focus on the lemmatized form of common nouns
ROMANCE_LOAN_NOUNS = [
    "denbora", "liburu", "kolore", "eliza", "festa", "bakea", 
    "milioi", "mundua", "historia", "plaza", "arraza"
]

NATIVE_BASQUE_NOUNS = [
    "etxe", "txakur", "zuhaitz", "mendia", "harria", "bihotz", 
    "eguna", "itsasoa", "lurra", "jendea", "umea"
]

def analyze_lexical_sets(word_list, list_name, model, tokenizer, typo_vector, device):
    """
    Runs inference on a list of words and calculates average routing weights for ALL experts.
    """
    aggl_weights = []
    indo_weights = []
    romance_weights = []
    germ_weights = []
    
    print(f"\nAnalyzing '{list_name}' Lexical Set ({len(word_list)} words)...")
    model.eval()
    
    with torch.no_grad():
        for word in word_list:
            inputs = tokenizer([word], is_split_into_words=True, return_tensors="pt").to(device)
            typo_vec_expanded = typo_vector.unsqueeze(0).to(device)
            
            _, routing_weights = model(inputs['input_ids'], inputs['attention_mask'], typo_vec_expanded)
            
            weights = routing_weights.squeeze(0)[0].cpu().numpy()
            
            # Record ALL 4 weights
            aggl_weights.append(weights[0])
            indo_weights.append(weights[1])
            romance_weights.append(weights[2])
            germ_weights.append(weights[3])

    # Calculate averages
    avg_aggl = np.mean(aggl_weights) * 100
    avg_indo = np.mean(indo_weights) * 100
    avg_rom = np.mean(romance_weights) * 100
    avg_germ = np.mean(germ_weights) * 100
    
    print(f"[{list_name}] STATS (Must sum to ~100%):")
    print(f"  -> Agglutinative (Expert 0): {avg_aggl:.1f}%")
    print(f"  -> Indo-Aryan    (Expert 1): {avg_indo:.1f}%")
    print(f"  -> Romance       (Expert 2): {avg_rom:.1f}%")
    print(f"  -> Germanic      (Expert 3): {avg_germ:.1f}%")
    print("-" * 50)
    
    return avg_rom, avg_aggl

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Loan Word Study on: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', token=False)
    model = TypologyMoEForPOS(NUM_LABELS, NUM_EXPERTS, TYPO_VECTOR_SIZE).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    # Load Basque typo vector
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    basque_vector = typo_vectors["eus"]
    
    # --- Execute Study ---
    loan_romance_score, loan_aggl_score = analyze_lexical_sets(
        ROMANCE_LOAN_NOUNS, "Romance Loan Nouns", model, tokenizer, basque_vector, device
    )
    
    native_romance_score, native_aggl_score = analyze_lexical_sets(
        NATIVE_BASQUE_NOUNS, "Native Basque Nouns", model, tokenizer, basque_vector, device
    )
    
    # Final Research Conclusion
    print("\n=== Control Study Conclusion for Project Report ===")
    if loan_romance_score > native_romance_score:
        diff_multiplier = loan_romance_score / native_romance_score
        print(f"Hypothesis Proven: The dynamic router is prioritizing token lexical similarity over global language typology vectors when encountering known loan words.")
        print(f"The Romance expert was activated {diff_multiplier:.1f}x more frequently for Romance loan words than for native Basque words of the same lexical category (NOUN).")
    else:
        print("Hypothesis Refuted: The dynamic router prioritizes typological vectors even for loan words.")

if __name__ == "__main__":
    main()
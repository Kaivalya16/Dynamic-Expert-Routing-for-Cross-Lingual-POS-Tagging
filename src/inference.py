import torch
from transformers import AutoTokenizer
from model import TypologyMoEForPOS
from data_loader import NUM_LABELS, TAG2ID

# 1. Configuration 
NUM_EXPERTS = 4
TYPO_VECTOR_SIZE = 65
EXPERT_NAMES = ["Agglutinative", "Indo-Aryan", "Romance", "Germanic"]

# Create a reverse dictionary to turn integer IDs back into readable tags like 'NOUN'
ID2TAG = {idx: tag for tag, idx in TAG2ID.items()}

def predict_sentence(sentence, model, tokenizer, typo_vector, device):
    # Split the raw string into a list of words
    words = sentence.split()
    
    # Tokenize the words, keeping track of how they are split into subwords
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt").to(device)
    word_ids = inputs.word_ids() # Tells us which original word each subword belongs to
    
    # Expand the typo vector for the batch size of 1
    typo_vec_expanded = typo_vector.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits, routing_weights = model(inputs['input_ids'], inputs['attention_mask'], typo_vec_expanded)
        
        # Get the highest probability prediction for each subword
        predictions = torch.argmax(logits, dim=-1).squeeze(0) # Shape: [sequence_length]
        weights = routing_weights.squeeze(0) # Shape: [sequence_length, 4]
        
    # We only want to print the prediction for the FIRST subword of every actual word
    print(f"\nAnalyzing Sentence: '{sentence}'")
    print("-" * 85)
    print(f"{'WORD':<15} | {'PREDICTED POS':<13} | {'ROUTING WEIGHTS (Aggl / Indo / Rom / Germ)'}")
    print("-" * 85)
    
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        # Ignore special tokens (<s>, </s>) and ignore subsequent subwords of a word
        if word_idx is not None and word_idx != previous_word_idx:
            word = words[word_idx]
            predicted_tag = ID2TAG[predictions[idx].item()]
            
            # Extract weights and format them nicely
            w_aggl = weights[idx][0].item() * 100
            w_indo = weights[idx][1].item() * 100
            w_rom  = weights[idx][2].item() * 100
            w_germ = weights[idx][3].item() * 100
            
            weight_string = f"{w_aggl:>5.1f}% / {w_indo:>5.1f}% / {w_rom:>5.1f}% / {w_germ:>5.1f}%"
            
            print(f"{word:<15} | {predicted_tag:<13} | {weight_string}")
            
        previous_word_idx = word_idx
    print("-" * 85)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Tokenizer and Model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', token=False)
    
    model = TypologyMoEForPOS(
        num_labels=NUM_LABELS, 
        num_experts=NUM_EXPERTS, 
        typo_vector_size=TYPO_VECTOR_SIZE
    ).to(device)
    
    # Make sure to load the Phase 2 unfrozen model!
    model.load_state_dict(torch.load("./checkpoints/typology_moe_unfrozen.pt"))
    
    # 3. Load the Basque Typology Vector
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    basque_vector = typo_vectors["eus"]
    
    # 4. Test Sentences
    # Here are a few basic Basque sentences to test. 
    # Feel free to add any Basque sentence you want to this list!
    test_sentences = [
        "Katuak esnea edaten du", # The cat drinks milk (Katuak=Noun, esnea=Noun, edaten=Verb, du=Aux)
        "Gizon handiak txakurra ikusi zuen", # The big man saw the dog (Gizon=Noun, handiak=Adj, txakurra=Noun, ikusi=Verb, zuen=Aux)
        "Gaur euria ari du" # It is raining today (Gaur=Adv, euria=Noun, ari=Verb, du=Aux)
    ]
    
    for sentence in test_sentences:
        predict_sentence(sentence, model, tokenizer, basque_vector, device)

if __name__ == "__main__":
    main()
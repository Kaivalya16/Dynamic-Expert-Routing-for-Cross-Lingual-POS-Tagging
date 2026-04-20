def analyze_lexical_sets_rl(word_list, list_name, model, tokenizer, typo_vector, device):
    """
    Runs inference on isolated words to see which expert the RL Agent STRICTLY chooses.
    """
    expert_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print(f"\nAnalyzing '{list_name}' Lexical Set ({len(word_list)} words)...")
    model.eval()
    
    with torch.no_grad():
        for word in word_list:
            inputs = tokenizer([word], is_split_into_words=True, return_tensors="pt").to(device)
            typo_vec_expanded = typo_vector.unsqueeze(0).to(device)
            
            # sample=False forces the RL agent to take its most confident action
            _, _, actions = model(inputs['input_ids'], inputs['attention_mask'], typo_vec_expanded, sample=False)
            
            # The action chosen for the first subword token
            chosen_expert = actions.squeeze(0)[0].item()
            expert_counts[chosen_expert] += 1

    total_words = len(word_list)
    aggl_pct = (expert_counts[0] / total_words) * 100
    indo_pct = (expert_counts[1] / total_words) * 100
    rom_pct = (expert_counts[2] / total_words) * 100
    germ_pct = (expert_counts[3] / total_words) * 100
    
    print(f"[{list_name}] RL HARD-ROUTING STATS:")
    print(f"  -> Agglutinative (Expert 0): {aggl_pct:.1f}% of words")
    print(f"  -> Indo-Aryan    (Expert 1): {indo_pct:.1f}% of words")
    print(f"  -> Romance       (Expert 2): {rom_pct:.1f}% of words")
    print(f"  -> Germanic      (Expert 3): {germ_pct:.1f}% of words")
    print("-" * 50)
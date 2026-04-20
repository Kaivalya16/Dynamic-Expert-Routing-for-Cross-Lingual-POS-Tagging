import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", 
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

TAG2ID = {tag: idx for idx, tag in enumerate(UPOS_TAGS)}
ID2TAG = {idx: tag for tag, idx in TAG2ID.items()} # Add this line!
NUM_LABELS = len(UPOS_TAGS)

class CrossLingualPOSDataset(Dataset):
    def __init__(self, conllu_file_path, lang_code, typo_vectors, tokenizer_name='xlm-roberta-base', max_len=128):
        self.sentences = [] # List of lists of words
        self.labels = []    # List of lists of POS tags
        self.lang_code = lang_code
        self.typo_vector = typo_vectors[lang_code]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=False)
        self.max_len = max_len
        
        # 1. TODO: Write your .conllu parsing logic here to populate self.sentences and self.labels
        # (Using a library like 'conllu' makes this a breeze)
        self._parse_conllu(conllu_file_path)

        # 2. Build a tag-to-ID mapping (e.g., {'NOUN': 0, 'VERB': 1, ...})
        self.tag2id = TAG2ID

    def _parse_conllu(self, filepath):
        """
        Parses a CoNLL-U file to extract sentences and their corresponding UPOS tags.
        Populates self.sentences and self.labels.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            current_sentence = []
            current_labels = []

            for line in f:
                line = line.strip()
                
                # 1. Skip metadata/comment lines
                if line.startswith('#'):
                    continue
                
                # 2. A blank line indicates the end of the current sentence
                if not line:
                    if current_sentence:
                        self.sentences.append(current_sentence)
                        self.labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                    continue
                
                # 3. Parse the tab-separated columns
                parts = line.split('\t')
                
                # CoNLL-U format strictly has 10 columns
                if len(parts) == 10:
                    token_id = parts[0]
                    
                    # Skip multi-word tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
                    if '-' in token_id or '.' in token_id:
                        continue
                        
                    word = parts[1]
                    upos = parts[3] # UPOS (Universal Part-of-Speech) is always the 4th column
                    
                    current_sentence.append(word)
                    current_labels.append(upos)
                    
            # Catch the very last sentence just in case the file doesn't end with a blank line
            if current_sentence:
                self.sentences.append(current_sentence)
                self.labels.append(current_labels)

    def _build_tag_vocab(self):
        """
        Iterates through all extracted labels to build a tag-to-ID integer mapping.
        """
        unique_tags = set()
        for labels in self.labels:
            unique_tags.update(labels)
            
        # Sort the tags alphabetically so the mapping remains consistent across runs
        tag2id = {tag: idx for idx, tag in enumerate(sorted(list(unique_tags)))}
        
        print(f"[{self.lang_code}] Extracted {len(self.sentences)} sentences.")
        print(f"[{self.lang_code}] Found {len(tag2id)} unique POS tags.")
        
        return tag2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        # Tokenize words and align labels
        tokenized_inputs = self.tokenizer(words, is_split_into_words=True, 
                                          padding='max_length', truncation=True, 
                                          max_length=self.max_len, return_tensors="pt")
        
        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like <s> and </s> get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the actual POS tag ID
                label_ids.append(self.tag2id[tags[word_idx]])
            else:
                # Subsequent subwords get -100
                label_ids.append(-100)
            previous_word_idx = word_idx

        # Squeeze out the batch dimension added by the tokenizer
        item = {key: val.squeeze(0) for key, val in tokenized_inputs.items()}
        item['labels'] = torch.tensor(label_ids)
        item['typo_vecs'] = self.typo_vector # Inject the language feature!
        
        return item
    
from torch.utils.data import DataLoader

def get_dataloaders(filepath, lang_code, typo_vectors, batch_size=32, tokenizer_name='xlm-roberta-base'):
    """
    Creates and returns a DataLoader for a given dataset.
    Returns (None, dataloader) to match the unpack signature in the training script.
    """
    dataset = CrossLingualPOSDataset(
        conllu_file_path=filepath, 
        lang_code=lang_code, 
        typo_vectors=typo_vectors, 
        tokenizer_name=tokenizer_name
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return None, dataloader
    

if __name__ == "__main__":
    # Load the typology vectors you generated earlier
    typo_vectors = torch.load("./data/processed/typology_vectors.pt")
    
    # Point this to one of your downloaded train.conllu files
    # For example, the English EWT dataset
    sample_file = "./data/raw/UD_English-EWT-master/en_ewt-ud-train.conllu"
    
    # Initialize the dataset
    dataset = CrossLingualPOSDataset(
        conllu_file_path=sample_file, 
        lang_code="eng", 
        typo_vectors=typo_vectors
    )
    
    # Fetch the very first item
    sample_item = dataset[0]
    
    print("\n--- Testing Dataset Output ---")
    print("Tokens Shape:", sample_item['input_ids'].shape)
    print("Labels Shape:", sample_item['labels'].shape)
    print("Typology Vector Shape:", sample_item['typo_vector'].shape)
    
    # Decode the first few tokens to visually verify alignment
    print("\nToken -> Label ID Alignment:")
    tokens = dataset.tokenizer.convert_ids_to_tokens(sample_item['input_ids'][:15])
    labels = sample_item['labels'][:15].tolist()
    
    for token, label in zip(tokens, labels):
        safe_token = token.replace('\u2581', '_')
        print(f"{safe_token:15} -> {label}")
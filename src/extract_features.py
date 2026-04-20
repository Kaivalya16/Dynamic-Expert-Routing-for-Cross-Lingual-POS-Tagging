import lang2vec.lang2vec as l2v
import torch
import os

# ISO 639-3 codes for our 9 languages
# Target: Basque (eus). Source: Turkish (tur), Finnish (fin), Hindi (hin), Marathi (mar), Spanish (spa), French (fra), English (eng), German (deu)
# Note: lang2vec uses 3-letter codes
languages = ["eus", "tur", "fin", "hin", "mar", "spa", "fra", "eng", "deu"]


def get_typological_vectors():
    print("Fetching URIEL features...")
    # We pull syntax (word order) and morphology (fusion/agglutination) features
    # 'knn' versions fill in missing data points based on nearest typological neighbors
    features = l2v.get_features(languages, "syntax_knn", "phonology_knn", "inventory_knn")
    
    # Convert to a dictionary of PyTorch tensors
    vector_dict = {}
    for lang in languages:
        # Concatenate the feature arrays for each language into one long vector
        vector = features[lang] 
        vector_dict[lang] = torch.tensor(vector, dtype=torch.float32)
        print(f"{lang} vector shape: {vector_dict[lang].shape}")
        
    # Save the dictionary to the processed data folder
    os.makedirs("./data/processed", exist_ok=True)
    torch.save(vector_dict, "./data/processed/typology_vectors.pt")
    print("Saved typology vectors to data/processed/typology_vectors.pt")

if __name__ == "__main__":
    get_typological_vectors()
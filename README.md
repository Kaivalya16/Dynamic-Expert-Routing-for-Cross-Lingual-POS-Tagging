# TyMoE: Zero-Shot Cross-Lingual POS Tagging using Typology-Conditioned Mixture of Experts

**TyMoE** is a custom neural architecture designed to solve the "lexical bias" problem in zero-shot cross-lingual NLP. Standard multilingual models often rely on superficial vocabulary overlap, failing when applied to language isolates (like Basque) that have unique grammar but share loan words with neighboring languages.

This project introduces a **Typology-Conditioned Mixture of Experts (MoE)**. By feeding continuous URIEL typological vectors (capturing syntax and morphology rules) into a dynamic gating network, the model learns to route tokens to specialized Language-Family Experts based on deep structural grammar, rather than just vocabulary.

## 🚀 Key Achievements

  * **Zero-Shot Language Isolate Transfer:** Achieved **70.96%** Part-of-Speech (UPOS) tagging accuracy on Basque (Eus) without a single Basque training sentence.
  * **Interpretable Routing:** Proved that the model learns linguistic correlations. For example, the Router dynamically sent **97.5% of Basque Verbs** to the Indo-Aryan expert, independently discovering the shared strictly SOV (Subject-Object-Verb) and ergative structures between Basque and Hindi/Marathi.
  * **Mechanistic Probing:** Demonstrated via controlled lexical studies that the Router utilizes deep contextual embeddings to override pre-trained geographical biases when handling Romance loan words in Basque text.

## 🧠 Architecture Overview

Our architecture modifies a standard `xlm-roberta-base` backbone with a custom MoE head:

1.  **Dual-Path Input:** For each token, the model processes the contextual Token Embedding (768-d) and the global Language Typology Vector (65-d).
2.  **Dynamic Gating Network (Router):** Evaluates the token context alongside the typological rules to output a Softmax distribution of trust weights.
3.  **Language-Family Experts:** Four parallel linear classifiers specialized in high-resource language typologies:
      * Agglutinative (Turkish, Finnish)
      * Indo-Aryan (Hindi, Marathi)
      * Romance (Spanish, French)
      * Germanic (English, German)
4.  **Weighted Aggregation:** Expert predictions are scaled by the router weights to produce the final 17-class UPOS prediction.

### Two-Phase Training Strategy

To prevent catastrophic forgetting of the pre-trained XLM-R embeddings, training is split:

  * **Phase 1 (Warm-up):** The XLM-R backbone is **frozen**. Only the Router and Experts are trained. We apply a custom **Load Balancing Loss** to penalize the router if it collapses and relies on only one expert.
  * **Phase 2 (Full Fine-Tuning):** The entire network is **unfrozen** with a reduced learning rate ($1e-5$), allowing the deep contextual embeddings to structural align with our typological routing mechanism.

## 📂 Project Structure

```text
├── data/
│   ├── raw/                 # Raw .conllu Universal Dependency files
│   └── processed/           # Cached subword alignments and typology_vectors.pt
├── notebooks/               # Jupyter notebooks for EDA and routing visualizations
├── src/
│   ├── extract_features.py  # Queries lang2vec for URIEL typology vectors
│   ├── data_loader.py       # Parses .conllu and handles XLM-R subword alignment
│   ├── model.py             # Defines the TyMoE PyTorch architecture
│   ├── train.py             # Phase 1: Frozen backbone training
│   ├── train_unfrozen.py    # Phase 2: Unfrozen full fine-tuning
│   ├── evaluate_unfrozen.py # Calculates zero-shot accuracy & aggregates routing weights
│   ├── inference.py         # Token-by-token qualitative testing on raw sentences
│   └── loan_word_study.py   # Mechanistic probing of router behavior on lexical sets
├── requirements.txt         # Environment dependencies
└── README.md
```

## ⚙️ Installation & Setup

1.  Clone the repository:

<!-- end list -->

```bash
git clone https://github.com/yourusername/TyMoE-CrossLingual-POS.git
cd TyMoE-CrossLingual-POS
```

2.  Create a virtual environment and install dependencies:

<!-- end list -->

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3.  Ensure you have the required UD Treebank `.conllu` files in the `data/raw/` directory.

## 💻 Usage

**1. Extract Typological Features**

```bash
python src/extract_features.py
```

**2. Train the Model**
Run Phase 1 (Frozen), followed by Phase 2 (Unfrozen):

```bash
python src/train.py
python src/train_unfrozen.py
```

**3. Evaluate Zero-Shot Accuracy**
Test the model on your target language isolate (Basque) and view the aggregated routing distributions per Part-of-Speech:

```bash
python src/evaluate_unfrozen.py
```

**4. Real-time Inference**
Pass raw Basque sentences through the model to see token-by-token routing decisions:

```bash
python src/inference.py
```

**5. Run the Probing Experiment**
Run the mechanistic interpretability study comparing how the router handles Romance loan words versus Native Basque words in isolation:

```bash
python src/loan_word_study.py
```

## 📝 Acknowledgements

  * **Universal Dependencies:** For providing the standardized cross-lingual treebanks.
  * **URIEL & lang2vec:** For the linguistic typology databases that power the routing conditioning.
  * **HuggingFace:** For the `xlm-roberta-base` transformer backbone.
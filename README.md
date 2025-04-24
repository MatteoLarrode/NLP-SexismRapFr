# "Balance Ton Rap": Quantifying gender stereotypes in French rap using word embeddings

This repository contains a Natural Language Processing (NLP) project focused on quantifying gender bias in French rap lyrics using word embedding association tests (WEAT). By training static embeddings on a corpus of 8,208 French rap songs and comparing them with off-the-shelf embeddings trained on general French text, I examine differences in stereotypical gender associations across four conceptual categories. 

## Repository structure

The repository is organized as follows:

```
NLP-SexismRapFr/
├── data/                  # Data directory
│   ├── analogy_dataset.pkl          # Dataset for word analogy tasks
│   ├── french_rap_lyrics_raw.pkl    # Raw collected lyrics
│   ├── processed_lyrics_lemmatized.pkl  # Lemmatized lyrics
│   ├── processed_lyrics.pkl         # Processed lyrics
│   └── wordlist353-fr.csv           # French word similarity evaluation list
│
├── figs/                  # Figures and visualizations
│   ├── gendered_words_similarity/   # Gender-related word similarity analysis
│   ├── k_means/                     # K-means clustering visualizations
│   ├── similarity_corr/             # Word similarity correlation plots
│   └── WEAT/                        # Word Embedding Association Tests
│
├── utils/                 # Utility scripts
│   ├── analysis_helpers.py         # Analysis utility functions
│   ├── model_download_helpers.py   # Functions to download models
│   ├── preprocessing_helpers.py    # Text preprocessing utilities
│   ├── training_helpers.py         # Training utility functions
│   ├── validation_helpers.py       # Validation utility functions
│   ├── visualisations_helpers.py   # Visualization helpers
│   └── word_choice_helpers.py      # Word selection utilities
│
├── 01_data_collection_preprocessing.ipynb  # Data collection and preprocessing
├── 02_model_training.ipynb                  # Model training notebook
├── 03_word_embeddings_validation.ipynb      # Validation of word embeddings
├── 04_gender_bias_analysis.ipynb            # Analysis of gender bias
│
└── environment.yml        # Conda environment specification
```
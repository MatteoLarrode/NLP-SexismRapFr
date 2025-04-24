# "Balance Ton Rap": Quantifying gender stereotypes in French rap using word embeddings

This repository contains a Natural Language Processing (NLP) project focused on quantifying gender bias in French rap lyrics using word embedding association tests (WEAT). By training static embeddings on a corpus of 8,208 French rap songs and comparing them with off-the-shelf embeddings trained on general French text, I examine differences in stereotypical gender associations across four conceptual categories. 

## Repository structure

The repository is organized as follows:

```
NLP-SexismRapFr/
├── data/                  # Data directory
│   ├── analogy_dataset.pkl          # Dataset for word analogy tasks
│   ├── french_rap_lyrics_raw.pkl    # Raw lyrics from Zurbuchen & Voigt (2024) -- https://github.com/ljz112/CLResearch
│   ├── processed_lyrics_lemmatized.pkl  # Lemmatized lyrics 
│   ├── processed_lyrics.pkl         # Processed lyrics
│   └── wordlist353-fr.csv           # French word similarity evaluation list
│
├── figs/                  # Figures and visualisations
│   ├── gendered_words_similarity/   # Word similarity complementary analysis of words with different gendered forms
│   ├── k_means/                     # K-means clustering visualiation
│   ├── similarity_corr/             # Word similarity correlation plots
│   └── WEAT/                        # Word Embedding Association Tests
│
├── utils/                 # Utility scripts
│   ├── analysis_helpers.py         # Analysis utility functions and class (mostly WEAT)
│   ├── model_download_helpers.py   # Functions to download off-the-shelf models from Fauconnier (2015) - http://fauconnier.github.io
│   ├── preprocessing_helpers.py    # Text preprocessing utilities
│   ├── training_helpers.py         # Training utility functions
│   ├── validation_helpers.py       # Word embeddings validation utility functions
│   ├── visualisations_helpers.py   # Visualisation helpers
│   └── word_choice_helpers.py      # Word selection (for attribute and target sets) utilities
│
├── notebooks/
│   ├── 01_data_collection_preprocessing.ipynb  # Data collection and preprocessing, and downloading of off-the-shelf embeddings
│   ├── 02_model_training.ipynb                  # Model training (creation of word embeddings)
│   ├── 03_word_embeddings_validation.ipynb      # Validation of word embeddings
│   ├── 04_gender_bias_analysis.ipynb            # Analysis of gender bias
│
└── environment.yml        # Conda environment dependencies
```
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
│   └── 04_gender_bias_analysis.ipynb            # Analysis of gender bias
│
└── environment.yml        # Conda environment dependencies
```

## Installation

To set up the project environment:

```
# Clone the repository
git clone [link to this github page]
cd NLP-SexismRapFr

# Create conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate nlp-summative
```

## Replication

The entire study, including visualisations and analysis, is replicable step-by-step by simply using the notebooks.
The models were too large to be uploaded to Github, so the notebooks offer all the necessary code to train them in the same conditions as in the study.

1. Start with `01_data_collection_preprocessing.ipynb`. The rap lyrics cleaning is optional, as the preprocessed corpus was pickled in `data/processed_lyrics.pkl`. However, the downloading of off-the-shelf embeddings of general French text is necessary for the replication. 

2. Continue with `02_model_training.ipynb` to train the word embeddings, using Word2Vec, on the processed rap lyrics corpus.

3. Explore `03_word_embeddings_validation.ipynb` to replicate the evaluation of all embeddings used in the project.

4. Replicate the WEAT experiment in `04_gender_bias_analysis.ipynb`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
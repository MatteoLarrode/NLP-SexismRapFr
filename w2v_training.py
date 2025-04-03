import wandb
from wandb.keras import WandCallbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import nltk
from nltk.tokenize import word_tokenize
import re

wandb.init(
    project="french-rap-word-embeddings",
    name=f"word2vec-{timestamp}",
    config={
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "iterations": iterations,
        "seed": seed
    }
)

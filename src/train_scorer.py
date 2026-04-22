"""
Run this once to train and save the NN scorer:
    python -m src.train_scorer
"""
from src.nn_scorer import train

if __name__ == "__main__":
    train(model_path="data/scorer_model.pkl")

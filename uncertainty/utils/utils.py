"""Minimal utility functions."""
import os
import pickle
from argparse import ArgumentParser
from uncertainty.models.huggingface_models import HuggingfaceModel


def init_model(args):
    """Initialize the Huggingface model."""
    return HuggingfaceModel(args.model_name, max_new_tokens=args.model_max_new_tokens)


def save(obj, file, save_dir):
    """Save object to file in specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Saved] {file} → {path}")


def get_parser():
    """Return a minimal argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b")
    parser.add_argument("--model_max_new_tokens", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser

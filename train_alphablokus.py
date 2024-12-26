"""
train_alphablokus.py

Usage:
  python train_alphablokus.py [--device cpu/cuda] [--iterations 10] ...
"""

import argparse
import torch
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_agents import AlphaBlokusNet
from training.train_loop import train_alphablokus
from training.evaluator import evaluate_vs_random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total_iterations", type=int, default=1)
    parser.add_argument("--games_per_iteration", type=int, default=1)
    parser.add_argument("--n_mcts_sim", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_steps_per_iter", type=int, default=100)
    parser.add_argument("--replay_buffer_max", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_games", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create the model
    model = AlphaBlokusNet()

    # Train
    train_alphablokus(model,
                      device=args.device,
                      total_iterations=args.total_iterations,
                      games_per_iteration=args.games_per_iteration,
                      n_mcts_sim=args.n_mcts_sim,
                      batch_size=args.batch_size,
                      train_steps_per_iter=args.train_steps_per_iter,
                      replay_buffer_max=args.replay_buffer_max,
                      lr=args.lr)

    # Evaluate
    avg_score = evaluate_vs_random(model, n_games=args.eval_games, device=args.device)
    logging.info(f"Evaluation done. Avg Score color0 = {avg_score}")

    # Save model
    save_path = "alphablokus_model.pth"
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

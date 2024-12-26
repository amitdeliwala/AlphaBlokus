# training/train_loop.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from training.self_play import generate_self_play_data
from training.replay_buffer import ReplayBuffer

def train_alphablokus(model, 
                      device='cpu',
                      total_iterations=10,
                      games_per_iteration=10,
                      n_mcts_sim=1,
                      batch_size=32,
                      train_steps_per_iter=100,
                      replay_buffer_max=50000,
                      lr=1e-3):
    """
    A basic training loop that:
    1) For iteration in [1..total_iterations]:
       a) Generate self-play data from N games
       b) Store in replay buffer
       c) Train model for K steps
       d) Log progress

    We use an Adam optimizer. We track policy loss & value loss via logging.
    """

    logger = logging.getLogger("AlphaBlokusTrain")
    logger.setLevel(logging.INFO)

    replay_buffer = ReplayBuffer(max_size=replay_buffer_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for it in range(1, total_iterations+1):
        logger.info(f"=== Iteration {it} ===")
        # 1) Self-play
        model.eval()
        samples = generate_self_play_data(model, n_games=games_per_iteration, 
                                          n_mcts_sim=n_mcts_sim, device=device)
        logger.info(f"Generated {len(samples)} samples from self-play.")
        replay_buffer.push(samples)

        # 2) Training
        model.train()
        if len(replay_buffer) < batch_size:
            logger.info("Not enough samples in buffer to train.")
            continue

        for step in range(train_steps_per_iter):
            board_batch, piece_batch, pi_batch, value_batch = replay_buffer.sample(batch_size)
            board_batch = board_batch.to(device)
            piece_batch = piece_batch.to(device)
            pi_batch = pi_batch.to(device)
            value_batch = value_batch.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(board_batch, piece_batch)

            # policy loss (cross-entropy with pi_batch)
            # We use F.log_softmax for stability
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(pi_batch * log_probs, dim=1).mean()

            # value loss (MSE or huber)
            # We have value_batch as is (like - leftover squares).
            # Let's just do MSE
            value_loss = F.mse_loss(value_pred, value_batch)

            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

        # Log some info
        logger.info(f"Completed training steps. Policy loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}")

    logger.info("Training loop finished.")

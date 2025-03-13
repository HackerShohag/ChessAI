import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import board_to_matrix

class ChessAI:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []  # Replay buffer
        self.gamma = 0.99  # Discount factor

    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        best_move = None
        best_value = -float("inf")

        for move in legal_moves:
            board.push(move)
            state = torch.tensor(board_to_matrix(board), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            value = self.model(state).item()
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move

        return best_move if best_move is not None else random.choice(legal_moves)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, next_states, rewards = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        target_qs = rewards + self.gamma * self.model(next_states).detach().squeeze()
        predicted_qs = self.model(states).squeeze()

        loss = self.criterion(predicted_qs, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


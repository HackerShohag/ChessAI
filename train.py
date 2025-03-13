import pygame
import torch
import chess
import os

from AI import ChessAI
from model import ChessNet
from utils import draw_board, get_legal_moves, train_self_play

def main():
    # Initialize the chess AI and model
    model = ChessNet()
    ai = ChessAI(model)
    
    # Set the number of self-play games and training iterations
    num_games = 1000
    num_iterations = 10
    
    # Train the model using self-play
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        train_self_play(ai, num_games)
        torch.save(model.state_dict(), f"chess_model_{iteration + 1}.pth")
        print(f"Model saved after iteration {iteration + 1}")

if __name__ == "__main__":
    main()
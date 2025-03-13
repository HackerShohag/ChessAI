import pygame
import torch
import chess
import os

from AI import ChessAI
from model import ChessNet
from utils import draw_board, get_legal_moves

# Pygame setup
WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
WHITE, BLACK = (238, 238, 210), (118, 150, 86)
HIGHLIGHT = (186, 202, 68)
LEGAL_MOVE_COLOR = (255, 255, 0, 100)  # Semi-transparent yellow for legal moves

# Load model
model = ChessNet()
model_path = "chess_ai.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

ai = ChessAI(model)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI")

    board = chess.Board()
    selected_square = None
    legal_moves = []  # List of legal moves for the selected piece

    running = True
    while running:
        screen.fill((0, 0, 0))
        draw_board(screen, board, selected_square, legal_moves)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and board.turn:
                x, y = event.pos
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                square = chess.square(col, row)

                if selected_square is None:
                    if board.piece_at(square):  
                        selected_square = square
                        legal_moves = get_legal_moves(board, square)
                else:
                    if square in legal_moves:
                        move = chess.Move(selected_square, square)
                        board.push(move)
                        selected_square = None
                        legal_moves = []
                    else:
                        selected_square = None  

        if not board.turn and not board.is_game_over():
            pygame.time.wait(500)
            board.push(ai.select_move(board))

    # Save the model before quitting
    torch.save(model.state_dict(), model_path)
    pygame.quit()

# def get_legal_moves_from_ai(board, square):
#     """
#     Use the AI model to get the best legal moves for the selected piece.
#     """
#     legal_moves = [move.to_square for move in board.legal_moves if move.from_square == square]
#     best_move = ai.select_move(board)
    
#     if best_move and best_move.from_square == square:
#         legal_moves.append(best_move.to_square)

#     return legal_moves

if __name__ == "__main__":
    main()

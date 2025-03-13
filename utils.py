import chess
import chess.engine
import chess.pgn
from tqdm import tqdm
import numpy as np
import pygame

# draw_board function is defined below

# Pygame setup
# Pygame setup
WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
WHITE, BLACK = (238, 238, 210), (118, 150, 86)
HIGHLIGHT = (186, 202, 68)
LEGAL_MOVE_COLOR = (255, 255, 0, 100)  # Semi-transparent yellow

# Load chess pieces
piece_images = {}
piece_names = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
piece_theme = 'merida'

def board_to_matrix(board):
    # piece_map = {
    #     'wP': 1, 'wN': 2, 'wB': 3, 'wR': 4, 'wQ': 5, 'wK': 6,
    #     'bP': -1, 'bN': -2, 'bB': -3, 'bR': -4, 'bQ': -5, 'bK': -6, 
    #     None: 0
    # }
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6, 
        None: 0
    }
    matrix = np.zeros((8, 8), dtype=object)
    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        piece = board.piece_at(square)
        matrix[row][col] = piece_map[str(piece)] if piece else None

    matrix = np.array(matrix, dtype=np.float32)
    return matrix

for piece in piece_names:
    piece_images[piece] = pygame.transform.scale(
        pygame.image.load(f"piece/{piece_theme}/{piece}.svg"), (SQUARE_SIZE, SQUARE_SIZE)
    )

def draw_board(screen, board, selected_square, legal_moves):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.piece_at(chess.square(col, row))
            if piece:
                piece_image = pygame.image.load(f"piece/{piece_theme}/{piece}.svg")
                piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    if selected_square is not None:
        col, row = chess.square_file(selected_square), chess.square_rank(selected_square)
        pygame.draw.rect(screen, HIGHLIGHT, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    for move in legal_moves:
        col, row = chess.square_file(move), chess.square_rank(move)
        pygame.draw.circle(screen, LEGAL_MOVE_COLOR, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 4)

def get_legal_moves(board, position):
    if isinstance(position, str):
        position = chess.parse_square(position)

    print("Selected square:", chess.SQUARE_NAMES[position])
    legal_moves = [move.to_square for move in board.legal_moves if move.from_square == position]
    print("Legal moves:", [chess.SQUARE_NAMES[s] for s in legal_moves])

    return legal_moves

def train_self_play(ai, episodes=5000):
    for episode in tqdm(range(episodes)):
        board = chess.Board()
        states = []

        while not board.is_game_over():
            state = board_to_matrix(board)
            move = ai.select_move(board)

            if move is None:
                break  # No legal moves (checkmate/stalemate)
                
            board.push(move)
            states.append((state, board_to_matrix(board), 1 if board.turn else -1))  # Reward +1 for winning

        ai.memory.extend(states)
        ai.train()

def play_chess(ai, screen, board):
    selected_square = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and not board.turn:
                x, y = event.pos
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                square = row * 8 + col

                if selected_square is None:
                    selected_square = (row, col)
                else:
                    move = chess.Move(chess.square(selected_square[1], selected_square[0]), square)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                    else:
                        selected_square = None

        if board.turn and not board.is_game_over():
            pygame.time.wait(500)
            move = ai.select_move(board)
            if move:
                board.push(move)

        screen.fill((0, 0, 0))
        draw_board(screen, board, selected_square)
        pygame.display.flip()

    print("Game over:", board.result())
    pygame.quit()
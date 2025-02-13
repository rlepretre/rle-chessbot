import torch
import chess 
import numpy as np

def prepare_input(board: chess.Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

def board_to_matrix(board: chess.Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix
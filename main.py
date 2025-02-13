
# Load the mapping
import pickle
import torch
import numpy as np
import chess
from lib.chess_model import ChessModel
from lib.utils import prepare_input


from fastapi import FastAPI


with open("model/move_to_int", "rb") as file:
    move_to_int = pickle.load(file)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load the model
model = ChessModel(num_classes=len(move_to_int))
model.load_state_dict(torch.load("model/rle_100epochs.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode (it may be reductant)

int_to_move = {v: k for k, v in move_to_int.items()}


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/move_prediction/")
def predict_move(fen: str):

    board = chess.Board(fen)

    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return {"move": move, "fen": board.fen()}
    
    return None


        
    
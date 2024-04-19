"""
Setup the datasets for training.
"""

import os
import random
from pathlib import Path

import chess.engine
import chess.pgn
import tqdm


MAX_NUM_PIECES = 5

MULTIPLIERS = {
    "CAPTURE": 1,
    "CHECK": 1,
}


class ChessEngine:
    """Helper class to open and close the chess engine"""
    def __init__(self):
        stockfish_folder = Path("Dataset/stockfish")
        self.stockfish_file = next(stockfish_folder.iterdir())
    
    def __enter__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_file)
        return self.engine

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.quit()


def add_mate_in_two():
    """Adds the games from MateInTwo.pgn to the dataset"""
    pgn = open("Dataset/MateInTwo.pgn", encoding="latin-1")

    dataset = open("Dataset/dataset.pgn", "a+", encoding="latin-1")

    game = chess.pgn.read_game(pgn)
    found = 0
    added = 0

    while game:
        board = game.game().board()
        num_pieces = len(board.piece_map())
        if num_pieces == MAX_NUM_PIECES:
            print(game, file=dataset, end="\n\n")
            added += 1

        found += 1
        game = chess.pgn.read_game(pgn)

    print(f"Total games found: {found}")
    print(f"Total games added: {added}")

    dataset.close()
    pgn.close()

    return added


def add_lichess():
    """Adds the games from lichess_db_standard_rated_2016-01.pgn to the dataset"""
    pgn = open("Dataset/lichess_db_standard_rated_2016-01.pgn", encoding="latin-1")
    dataset = open("Dataset/dataset.pgn", "a+", encoding="latin-1")

    game = chess.pgn.read_game(pgn)
    found = 0
    added = 0

    with ChessEngine() as engine:
        with tqdm.tqdm(total=4_770_357) as pbar:
            while True:
                try:
                    curr_game = chess.pgn.read_game(pgn)
                except:
                    break

                found += 1
                pbar.update(1)

                if not curr_game:
                    break

                if "Normal" not in curr_game.headers.get("Termination", ""): continue

                # Get the final move and work backwards
                final_move = curr_game.end()

                # We only want checkmating games
                if not final_move.board().is_checkmate():
                    continue

                two_moves_before = final_move.parent.parent

                # Only <5 pieces for simplicity
                if len(two_moves_before.board().piece_map()) != MAX_NUM_PIECES:
                    continue

                analysis = engine.analyse(two_moves_before.board(), chess.engine.Limit(time=0.1))
                score = analysis["score"]
                checkmate = score.is_mate()
                # abs() because the score is negative if it's a checkmate to black
                mate_in_n = abs(score.relative.mate())

                # Checkmate in 2:
                if not checkmate or mate_in_n != 2:
                    continue
                
                # recreate the game
                
                game = chess.pgn.Game.without_tag_roster()
                game.setup(two_moves_before.board())

                line = two_moves_before.mainline_moves()
                game.add_line(line)

                print(game, file=dataset, end="\n\n")
                added += 1

                if not added % 100:
                    print(f"Added {added} games")

    print(f"Total games found: {found}")
    print(f"Total games added: {added}")

    dataset.close()
    pgn.close()

    return added

def train_test_split(count):
    """Splits the dataset into train and test sets"""
    train = open("Dataset/train.pgn", "w", encoding="latin-1")
    test = open("Dataset/test.pgn", "w", encoding="latin-1")
    dataset = open("Dataset/dataset.pgn", "r", encoding="latin-1")

    train_count = int(count * 0.8)
    train_indices = set(random.sample(range(count), train_count))

    for i in tqdm.tqdm(range(count)):
        game = chess.pgn.read_game(dataset)
        if i in train_indices:
            print(game, file=train, end="\n\n")
        else:
            print(game, file=test, end="\n\n")
    
    train.close()
    test.close()
    dataset.close()

if __name__ == "__main__":
    count = add_mate_in_two()
    count += add_lichess()
    train_test_split(count)
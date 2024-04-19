"""Runs a simple Q-learning algorithm on a chess engine."""

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.pgn
from setup_file import MULTIPLIERS, ChessEngine
from tqdm import tqdm

LEARNING_FACTORS = [0.1, 0.3, 0.5, 0.7, 0.9]


class ChessAgent:
    """A simple piece agent for Q-Learning."""
    def __init__(self, piece_type: int, learning_factor: float):
        self.piece_type = piece_type
        self.learning_factor = learning_factor
        self.q_table: dict[int, dict[int, float]] = {}

    @staticmethod
    def get_row(move: chess.Move) -> int:
        """Gets the row value for the Q-table from the move."""
        return move.from_square
    
    @staticmethod
    def get_column(board: chess.Board, move: chess.Move) -> int:
        """Gets the column value for the Q-table from the move."""
        square = move.to_square

        if board.is_capture(move):
            square += 64 * MULTIPLIERS["CAPTURE"]
        elif board.is_check():
            square += 64 * MULTIPLIERS["CHECK"]
        
        return square

    def bellman_equation(self, reward: int, row: int, col: int) -> float:
        """Bellman's Equation to update the Q-table."""
        r = self.q_table.setdefault(row, {}).setdefault(col, 0)
        return r + self.learning_factor * reward

    def update_value(self, board: chess.Board, move: chess.Move, reward: int):
        """Updates the Q-table with the new value."""
        row = self.get_row(move)
        col = self.get_column(board, move)
        self.q_table[row][col] = self.bellman_equation(reward, row, col)
    

class MasterAgent:
    """A master agent that controls the individual agents."""
    def __init__(self):
        self.AGENTS: dict[float, dict[int, ChessAgent]] = {lf: {
            piece: ChessAgent(piece, lf) for piece in range(1, 7) # Pawns to King
        } for lf in LEARNING_FACTORS}

        # Used for rerunning interrupted training
        self.starting_values: dict[float, int] = {lf: 0 for lf in LEARNING_FACTORS}
        
    def make_move(self, board: chess.Board, learning_factor: float) -> chess.Move:
        """Makes a move based on the Q-table."""
        best_reward = float("-inf")
        best_move = None

        for move in board.legal_moves:
            piece = board.piece_at(move.from_square).piece_type
            agent = self.AGENTS[learning_factor][piece]

            row, col = agent.get_row(move), agent.get_column(board, move)
            if row not in agent.q_table:
                continue

            if col not in agent.q_table[row]:
                continue

            reward = agent.q_table[row][col]

            # Multiply reward tenfold for checkmating movies

            new_board = board.copy()
            new_board.push(move)
            if new_board.is_checkmate():
                reward *= 10

            if reward > best_reward:
                best_reward = reward
                best_move = move

        if not best_move:
            best_move = random.choice(list(board.legal_moves))
        
        return best_move

    #region Saving and Loading
    @classmethod
    def from_cache(cls):
        """Get Master Agent from cache - due to interruptions."""
        self = cls()
        for lf in LEARNING_FACTORS:
            cache_folder = Path(f"cache/{lf}")
            if not cache_folder.exists():
                continue

            with open(cache_folder / "data.json") as f:
                data = json.load(f)
                self.starting_values[lf] = data["count"]
            
            for piece, agent in self.AGENTS[lf].items():
                cache_file = cache_folder / f"{piece}.json"
                with open(cache_file) as f:
                    agent.q_table = json.load(f)

        return self
    
    def to_cache(self, learning_factor: float, count: int):
        """Save the Master Agent to cache - for interruptions."""
        cache_folder = Path(f"cache/{learning_factor}")
        cache_folder.mkdir(parents=True, exist_ok=True)

        with open(cache_folder / "data.json", "w") as f:
            json.dump({"count": count}, f)

        for piece, agent in self.AGENTS[learning_factor].items():
            cache_file = cache_folder / f"{piece}.json"
            with open(cache_file, "w") as f:
                json.dump(agent.q_table, f)

    @classmethod
    def from_final(cls):
        """Get Master Agent from final training."""
        self = cls()
        results_folder = Path("results")

        for lf in LEARNING_FACTORS:
            for piece, agent in self.AGENTS[lf].items():
                res_file = results_folder / str(lf) / f"{piece}.json"
                with open(res_file) as f:
                    agent.q_table = json.load(f)

        return self
    
    def to_final(self):
        """Save the Master Agent to final training."""
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True)

        for lf in LEARNING_FACTORS:
            lf_folder = results_folder / str(lf)
            lf_folder.mkdir(exist_ok=True)

            for piece, agent in self.AGENTS[lf].items():
                res_file = lf_folder / f"{piece}.json"
                with open(res_file, "w") as f:
                    json.dump(agent.q_table, f)
    
    @classmethod
    def from_iter(cls, iteration: int, learning_factor: float):
        """Get Master Agent from a specific iteration."""
        self = cls()
        results_folder = Path("mid-results") / str(learning_factor) / str(iteration)

        for piece, agent in self.AGENTS[learning_factor].items():
            res_file = results_folder / f"{piece}.json"
            with open(res_file) as f:
                agent.q_table = json.load(f)

        return self
    
    def to_iter(self, iteration: int, learning_factor: float):
        """Save the Master Agent to a specific iteration."""
        results_folder = Path("mid-results") / str(learning_factor) / str(iteration)
        results_folder.mkdir(parents=True, exist_ok=True)

        for piece, agent in self.AGENTS[learning_factor].items():
            res_file = results_folder / f"{piece}.json"
            with open(res_file, "w") as f:
                json.dump(agent.q_table, f)
    
    #endregion

    def already_done(self, learning_factor: float, count: int) -> bool:
        """Check if training has already been done for this learning factor."""
        return count <= self.starting_values[learning_factor]

def main(learning_factor: float = None):
    """The main training loop for the Q-learning algorithm."""
    master_agent = MasterAgent.from_cache()
    training = open("Dataset/train.pgn", "r", encoding="latin-1")
    with open("Dataset/count.txt") as f:
        count = int(f.read())
    
    LFs = [learning_factor] if learning_factor else LEARNING_FACTORS
    for lf in LFs:
        if master_agent.already_done(lf, count):
            print(f"Skipping learning factor {lf}...")
            continue
        
        starting_value = master_agent.starting_values[lf]
        print(f"Training for learning factor {lf}...")
        print(f"Starting from Game #{starting_value}")
        training.seek(0)

        # Skip to the starting value
        for _ in range(starting_value): game = chess.pgn.read_game(training)

        with ChessEngine() as engine:
            for i in tqdm(range(starting_value, count)):
                game = chess.pgn.read_game(training)
                board = game.board()
                mainline_moves = game.mainline_moves()

                for j, move in enumerate(mainline_moves):
                    if (j % 2 == 1):
                        board.push(move)
                        continue
                
                    for legal_move in board.legal_moves:
                        piece = board.piece_at(legal_move.from_square).piece_type
                        agent = master_agent.AGENTS[lf][piece]

                        new_board = board.copy()
                        new_board.push(legal_move)

                        # Use stockfish to analyse the board
                        analysis = engine.analyse(new_board, chess.engine.Limit(time=0.1))
                        score = analysis["score"]
                        checkmate = score.is_mate()
                        mate_in_n = abs(score.relative.mate())
                        
                        # Reward system:
                        if checkmate and mate_in_n < 2:
                            reward = 1
                        elif checkmate:
                            reward = 0.75
                        elif board.is_capture(legal_move):
                            reward = 0.5
                        elif board.gives_check(legal_move):
                            reward = 0.1
                        elif new_board.is_stalemate():
                            reward = -1
                        else:
                            reward = -0.5

                        agent.update_value(board, legal_move, reward)

                    board.push(move)

                master_agent.to_cache(lf, i)

                if not i % 100:
                    print(f"Finished {i} games so far...")
                    master_agent.to_iter(i, lf)

    master_agent.to_final()

    training.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple Q-learning algorithm on a chess engine.")
    parser.add_argument("--lf", type=float, help="The learning factor to use.")

    args = parser.parse_args()
    if args.lf and args.lf not in LEARNING_FACTORS:
        print("Invalid learning factor. Please choose from 0.1, 0.3, 0.5, 0.7, 0.9.", file=sys.stderr)
    
    main(args.lf)




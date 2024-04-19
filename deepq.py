import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import chess
import chess.pgn
import torch
from setup_file import MULTIPLIERS, ChessEngine
from torch import nn, optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_row(move: chess.Move, board: chess.Board) -> torch.Tensor:
    """Get the row for the move in the board."""

    if board.is_capture(move) and board.gives_check(move):
        feature = 4
    elif board.gives_check(move):
        feature = 3
    elif board.is_capture(move):
        feature = 2
    else:
        feature = 1

    row = torch.zeros(64, device=device)
    row[move.from_square] = feature
    return row.unsqueeze(0)

def get_next_row(move: chess.Move, board: chess.Board) -> torch.Tensor:
    """Get the row for the next move in the board."""
    row = torch.zeros(64, device=device)
    row[move.to_square] = 1
    return row.unsqueeze(0)


class DeepQNetwork(nn.Module):
    """The Neural Network for each piece type."""
    def __init__(self):
        super().__init__()

        self.input_size = 64
        self.output_size = 64 * 3


        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 192)
        self.fc3 = nn.Linear(192, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, piece_type: int, boards: list[chess.Board]) -> torch.Tensor:

        qvals = self.fc1(x)
        qvals = self.relu(qvals)
        qvals = self.fc2(qvals)
        qvals = self.relu(qvals)
        qvals = self.fc3(qvals)

        mask = self.generate_mask(piece_type, boards)

        # Add a small value to avoid log(0)
        qvals = qvals + (mask + 1e-10).log()


        return qvals
    
    def generate_mask(self, piece_type: int, boards: list[chess.Board]) -> torch.Tensor:
        masks = torch.zeros((len(boards), self.output_size), device=device)

        for i, board in enumerate(boards):
            for move in board.legal_moves:
                if board.piece_at(move.from_square).piece_type != piece_type:
                    continue
                to_square = move.to_square
                if board.is_capture(move):
                    to_square += 64 * MULTIPLIERS["CAPTURE"]
                elif board.is_check():
                    to_square += 64 * MULTIPLIERS["CHECK"]
                
                masks[i, to_square] = 1

        return masks

@dataclass
class Transition:
    """Dataclass for the transitions in the replay memory."""
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    non_final: torch.Tensor
    board: chess.Board
    

class ReplayMemory:
    """Replay memory allows for sampling from memory rather than sequential learning."""
    def __init__(self, capacity: int):
        self.memory: deque[Transition] = deque(maxlen=capacity)
    
    def push(self, transition: Transition):
        self.memory.append(transition)
    
    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)




class MasterAgent:
    """The master agent that controls all the DeepQNetworks."""
    def __init__(self):
        self.AGENTS: dict[int, DeepQNetwork] = {
            chess.PAWN: DeepQNetwork().to(device),
            chess.KNIGHT: DeepQNetwork().to(device),
            chess.BISHOP: DeepQNetwork().to(device),
            chess.ROOK: DeepQNetwork().to(device),
            chess.QUEEN: DeepQNetwork().to(device),
            chess.KING: DeepQNetwork().to(device)
        }
    
    def make_move(self, board: chess.Board) -> chess.Move:
        best_val = float("-inf")
        best_move = None

        for move in board.legal_moves:
            piece = board.piece_at(move.from_square).piece_type

            to_square = move.to_square
            if board.is_capture(move):
                to_square += 64 * 1
            elif board.gives_check(move):
                to_square += 64 * 2
            agent = self.AGENTS[piece].eval()

            with torch.no_grad():
                state = get_row(move, board).to(device)
                qval = agent(state, piece, [board])[0][to_square]
            
                # l2 normalisation
                qval = qval / qval.norm()

            # Multiply tenfold if checkmate
            new_board = board.copy()
            new_board.push(move)
            if new_board.is_checkmate():
                qval *= 10
            
            if qval > best_val:
                best_val = qval
                best_move = move

        if best_move is None:
            best_move = random.choice(list(board.legal_moves))
        
        return best_move
    
    #region Saving and Loading
    @classmethod
    def from_final(cls):
        """Load the final weights of the agents."""
        self = cls()
        for piece in self.AGENTS:
            self.AGENTS[piece].load_state_dict(torch.load(f"weights/{piece}_dqn.pt"))
        return self

    def to_final(self):
        """Save the final weights of the agents."""
        Path("weights").mkdir(exist_ok=True)
        for piece, agent in self.AGENTS.items():
            torch.save(agent.state_dict(), f"weights/{piece}_dqn.pt")
        
    @classmethod
    def from_iter(cls, iteration: int):
        """Load the weights of the agents from a specific iteration."""
        self = cls()
        for piece in self.AGENTS:
            self.AGENTS[piece].load_state_dict(torch.load(f"mid-weights/{iteration}/{piece}_dqn.pt"))
        return self
    
    def to_iter(self, iteration: int):
        """Save the weights of the agents to a specific iteration."""
        Path(f"mid-weights/{iteration}").mkdir(parents=True, exist_ok=True)
        for piece, agent in self.AGENTS.items():
            torch.save(agent.state_dict(), f"mid-weights/{iteration}/{piece}_dqn.pt")

    #endregion


def main():
    """The main training loop"""

    master_agent = MasterAgent()
    target_agent = MasterAgent()
    CAPACITY = 2_000
    BATCH_SIZE = 64
    GAMMA = 0.9
    TAU = 5e-3
    LR = 1e-4

    optimisers = {
        piece: optim.Adam(agent.parameters(), lr=LR, amsgrad=True)
        for piece, agent in master_agent.AGENTS.items()
    }

    memory = {
        piece: ReplayMemory(CAPACITY)
        for piece in master_agent.AGENTS
    }

    training = open("training.pgn", "r", encoding="latin-1")
    with open("Dataset/count.txt", "r") as f:
        count = int(f.read())

    with ChessEngine() as engine:
        for i in tqdm(range(count)):
            game = chess.pgn.read_game(training)
            board = game.board()
            
            for j, move in enumerate(game.mainline_moves):
                if j % 2 == 0:
                    board.push(move)
                    continue
                for legal_move in board.legal_moves:

                    piece = board.piece_at(legal_move.from_square).piece_type

                    new_board = board.copy()
                    new_board.push(legal_move)

                    analysis = engine.analyse(new_board, chess.engine.Limit(time=0.1))
                    score = analysis["score"]
                    checkmate = score.is_mate()
                    mate_in_n = abs(score.relative.mate())

                    # Reward system:
                    if checkmate and mate_in_n < 2:
                        reward = 1
                    elif checkmate:
                        reward = -1
                    elif board.is_capture(move):
                        reward = 0.5
                    elif board.gives_check(move):
                        reward = 0.1
                    elif new_board.is_stalemate():
                        reward = -1
                    else:
                        reward = -0.5
                    
                    reward = torch.tensor([[reward]], device=device)

                    state = get_row(legal_move, board)
                    next_state = get_next_row(legal_move, board)
                    is_not_checkmate = torch.tensor([[not checkmate]], device=device)
                    action = torch.tensor([[legal_move.to_square]], device=device)
                    if board.is_capture(move):
                        action += 64 * MULTIPLIERS["CAPTURE"]
                    elif board.gives_check(move):
                        action += 64 * MULTIPLIERS["CHECK"]

                    # Add the transition to the memory
                    memory[piece].push(Transition(
                        state, action, reward,
                        next_state, is_not_checkmate,
                        board.copy(stack=False)
                    ))

                    for piece, mem in memory.items():
                        # The actual training section
                        if len(mem) < BATCH_SIZE:
                            continue

                        optimiser = optimisers[piece]
                        transitions = mem.sample(BATCH_SIZE)

                        not_checkmate_mask = torch.tensor(
                            [x.non_final[0][0] for x in transitions],
                            dtype=torch.bool,
                            device=device
                        )


                        next_state_batch = torch.cat([transition.next_state for transition in transitions if transition.non_final[0][0]])

                        # Grabs the batch of states, actions, rewards, and boards
                        state_batch = torch.cat([transition.state for transition in transitions])
                        action_batch = torch.cat([transition.action for transition in transitions])
                        reward_batch = torch.cat([transition.reward for transition in transitions])
                        boards = [transition.board for transition in transitions]
                        not_checkate_boards = [transition.board for transition in transitions if transition.non_final[0][0]]

                        agent = master_agent.AGENTS[piece]
                        target = target_agent.AGENTS[piece]

                        # Gets the Q values for the current state and the next state
                        qvals = agent(state_batch, piece, boards).gather(1, action_batch)
                        next_qvals = torch.zeros(BATCH_SIZE, device=device)
                        with torch.no_grad():
                            next_qvals[not_checkmate_mask] = target(next_state_batch, piece, not_checkate_boards).max(1).values
                            next_qvals = next_qvals.unsqueeze(1)
                        
                        target_qvals = reward_batch + GAMMA * next_qvals

                        criterion = nn.SmoothL1Loss()
                        loss = criterion(qvals, target_qvals)

                        optimiser.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 100)
                        optimiser.step()

                    for piece, agent in master_agent.AGENTS.items():
                        # Updates the target network - uses tau to slowly update the target network
                        state_dict = agent.state_dict()
                        target_state_dict = target_agent.AGENTS[piece].state_dict()

                        for key in state_dict:
                            target_state_dict[key] = TAU * state_dict[key] + (1 - TAU) * target_state_dict[key]
                        
                        target_agent.AGENTS[piece].load_state_dict(target_state_dict)

                board.push(move)

            if not i % 100:
                master_agent.to_iter(i)
        
        master_agent.to_final()
        training.close()

if __name__ == "__main__":
    main()


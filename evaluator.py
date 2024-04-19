"""Evaluates the agent's performance"""
import typing
from pathlib import Path

import chess
from setup_file import ChessEngine
from tqdm import tqdm


from deepq import MasterAgent as DeepQAgent, device
from simpleq import MasterAgent as SimpleQAgent

def get_average(make_move: typing.Callable[[chess.Board], chess.Move], averages: bool = True) -> float:
    """Get average time to checkmate"""
    testing = open("Dataset/test.pgn", "r", encoding="latin-1")

    game = chess.pgn.read_game(testing)
    times = []
    avg_time_to_checkmate = 0
    stalemates = 0
    tests = 0
    with ChessEngine() as engine:
        for _ in tqdm(range(100)):
            board = game.board()
            i = 0
            while not board.is_checkmate():
                if board.is_stalemate():
                    i = 20
                    stalemates += 1
                    break

                if (i % 2) == 0:
                    agent_move = make_move(board)
                    board.push(agent_move)
                else:
                    engine_move = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(engine_move.move)
            
                # If the game goes on for too long, break
                if i > 19:
                    break

                i += 1

            game = chess.pgn.read_game(testing)
            tests += 1
            avg_time_to_checkmate += i + 1
            times.append(i + 1)
                

    avg_time_to_checkmate /= tests
    print(f"{avg_time_to_checkmate} is the average time to checkmate")
    print(f"{stalemates} stalemates")
    print(f"Best checkmating time: {min(times)}")
    print(f"Times: {times}")

    if averages:
        return avg_time_to_checkmate
    else:
        return len([time for time in times if time < 21])

def get_average_simple(learning_factor: float, iteration: int = None, averages: bool = True) -> float:
    """Get average time to checkmate for SimpleQ agent"""

    master_agent = SimpleQAgent.from_final() if not iteration else SimpleQAgent.from_iter(iteration, learning_factor)
    return get_average(lambda board: master_agent.make_move(board, learning_factor), averages)

def get_average_deep(iteration: int = None, averages: bool = True) -> float:
    """Get average time to checkmate for DeepQ agent"""

    master_agent = DeepQAgent.from_final() if not iteration else DeepQAgent.from_iter(iteration)
    return get_average(master_agent.make_move, averages)

def get_average_random() -> float:
    """Get average time to checkmate for Random agent"""
    import random
    return get_average(lambda board: random.choice(list(board.legal_moves)))
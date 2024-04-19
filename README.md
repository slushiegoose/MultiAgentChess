# Can Chess Piece Agents with Imperfect Information Learn to Checkmate?

This repository contains the code for our coursework in COMP0124 Multi-Agent Artificial Intelligence at UCL. The project is a chess game where the agents have imperfect information about the board state. The agents are trained using reinforcement learning to play the game.

## Prerequisites

This project requires two datasets omitted from this repository due to their size. The datasets are available at the following links:

- [`Dataset/MateInTwo.pgn`](https://drive.google.com/file/d/1w_WI5O3apLk-Wg_dUzUqbs6LdvEZu-a_/view)
- [`Dataset/lichess_db_standard_rated_2016-01.pgn`](https://database.lichess.org/standard/lichess_db_standard_rated_2016-01.pgn.zst)

Please note that the second dataset is compressed and requires the `zstd` tool to decompress it. The tool can be installed using the following command:

```bash
sudo apt-get install zstd
```

Furthermore, a version of Stockfish is required to generate the training data. The Stockfish binary can be downloaded [here](https://stockfishchess.org/download/) and must be placed in the `Dataset/stockfish/` directory

Python 3.8 or later is required to run the code. The code has been tested on Python 3.10.0.

## Running

All of the Python packages required to run can be found in the `requirements.txt` file. To install the required packages, run the following command:

```bash
pip3 install -r requirements.txt
```

NOTE: All of the following code will be very time-consuming to run. We recommend running the code on a machine with a GPU.

To setup the training and testing datasets, run the following command:

```bash
python3 setup_file.py
```

To train the agents for Simple Q-Learning, run the following command:

```bash
python3 simpleq.py
```

To train the agents for Deep Q-Learning, run the following command:

```bash
python3 deepq.py
```

To plot the results of the training, run the following command:

```bash
python3 plotter.py
```

Note: `simpleq.py` has precautions for easy rerunning if the training is interrupted. `deepq.py` does not have this feature however, do to the necessity of replay memory and target networks.

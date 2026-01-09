# Learn2Slither - Snake with Q-Learning

A reinforcement learning project where a snake learns to play the classic snake game using Q-learning.

## Overview

This project implements a snake game where an AI agent learns optimal strategies through trial and error using Q-learning reinforcement learning.

## Features

- **10x10 game board** with snake, green apples (grow), and red apples (shrink)
- **Q-learning agent** that learns from experience
- **Graphical display** using pygame
- **Terminal vision output** showing what the snake sees
- **Model save/load** functionality
- **Configurable training parameters**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
```bash
./snake.py -sessions 100
```

### Training and Saving Model
```bash
./snake.py -sessions 1000 -save models/1000sess.pkl -visual off
```

### Loading and Evaluating Model
```bash
./snake.py -load models/1000sess.pkl -sessions 10 -dontlearn -visual on
```

### Step-by-Step Mode
```bash
./snake.py -load models/100sess.pkl -step-by-step -verbose
```

## Command Line Arguments

- `-sessions N` - Number of training sessions (default: 1)
- `-save PATH` - Save trained model to file
- `-load PATH` - Load pre-trained model
- `-visual on/off` - Enable/disable graphical display (default: on)
- `-speed slow/medium/fast/ultrafast` - Display speed (default: medium)
- `-step-by-step` - Enable step-by-step mode (press space to advance)
- `-dontlearn` - Disable learning (evaluation mode)
- `-board_size N` - Board size (default: 10)
- `-verbose` - Print state/vision during training

## Project Structure

```
learn2slither/
├── snake.py          # Main program
├── environment.py    # Game board and rules
├── agent.py          # Q-learning agent
├── display.py        # Pygame visualization
├── models/           # Saved models directory
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Requirements

- Python 3.7+
- numpy
- pygame

## License

Educational project for learning reinforcement learning concepts.

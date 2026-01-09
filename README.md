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

## How It Learns

### State Representation
The agent uses distance-based features from full vision (as shown in PDF):
- **Obstacle distance** (4 values) - Distance to wall/snake in each direction
- **Green apple distance** (4 values) - Distance to green apple (if visible) in each direction
- **Red apple distance** (4 values) - Distance to red apple (if visible) in each direction
- **Current direction** (1 value) - Snake's current facing direction

Total: 13 features encoding distances from what the snake sees in all 4 directions.

This respects the PDF requirement that the snake can see the full line in each direction, while creating a manageable state space (~10,000-100,000 states).

### Rewards
- Green apple: +10
- Red apple: -10
- Each move: -1
- Death (wall/self): -100

### Learning Parameters
- **Learning rate**: 0.2 (how much to update Q-values)
- **Discount factor**: 0.9 (importance of future rewards)
- **Epsilon decay**: 0.999 (exploration reduction rate)
- **Min epsilon**: 0.05 (always keep 5% exploration)

## Improvements Made

### 1. Fixed Red Apple Bug
Original code removed 2 tail segments instead of 1 when eating red apples.

### 2. Simplified State Space
Changed from full vision strings (millions of states) to 13 binary features (thousands of states).

### 3. Balanced Rewards
Adjusted rewards to be more balanced (was +50/-50, now +10/-10).

### 4. Tuned Hyperparameters
- Increased learning rate for faster learning
- Decreased epsilon decay for better exploration
- Lowered discount factor to prioritize immediate rewards

## Expected Performance

- **1 session**: Random movement, dies quickly
- **10 sessions**: Starts avoiding walls
- **100 sessions**: Learns to seek green apples
- **1000 sessions**: Achieves length 6-10 consistently
- **10000 sessions**: Achieves length 10-15+

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

from environment import Environment, Direction
from agent import QLearningAgent
from display import Display
import argparse
import gui
import sys
import os

# Import display only if needed
try:
    import pygame

    pygame.init()
    pygame.display.set_mode((1, 1))
    pygame.quit()
    DISPLAY_AVAILABLE = True
except (ImportError, pygame.error):
    DISPLAY_AVAILABLE = False


# ANSI color codes for terminal output
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"


def colorize_reward(reward):
    """Return colored reward string (red for negative, green for positive)

    Args:
        reward: Reward value

    Returns:
        str: Colored reward string
    """
    if reward < 0:
        return f"{COLOR_RED}{reward:.1f}{COLOR_RESET}"
    else:
        return f"{COLOR_GREEN}{reward:.1f}{COLOR_RESET}"


def get_safe_actions(env, state):
    """Get list of actions that don't lead to immediate death

    Args:
        env: Environment instance
        state: Current state

    Returns:
        list: Safe actions (won't hit wall or snake body immediately)
    """
    safe = []
    head_x, head_y = env.snake[0]

    for direction in Direction:
        # Get next position if we move in this direction
        if direction == Direction.UP:
            next_pos = (head_x, head_y - 1)
        elif direction == Direction.DOWN:
            next_pos = (head_x, head_y + 1)
        elif direction == Direction.LEFT:
            next_pos = (head_x - 1, head_y)
        else:  # RIGHT
            next_pos = (head_x + 1, head_y)

        # Check if safe (not wall, not snake body)
        is_safe = (
            0 <= next_pos[0] < env.board_size
            and 0 <= next_pos[1] < env.board_size
            and next_pos not in env.snake
        )

        if is_safe:
            safe.append(direction)

    return safe


def print_vision(state, action=None):
    """Print the snake's vision to terminal

    Args:
        state: State dictionary with vision
        action: Optional action taken
    """
    # Create the vision display
    up_vision = state["up"]
    down_vision = state["down"]
    left_vision = state["left"]
    right_vision = state["right"]

    # Build the cross pattern as shown in PDF
    print()  # Empty line before vision

    # Print up vision (each character on its own line)
    for char in up_vision:
        print(" " * len(left_vision) + char)

    # Print left + H + right vision (all on one line)
    print(left_vision + "H" + right_vision)

    # Print down vision (each character on its own line)
    for char in down_vision:
        print(" " * len(left_vision) + char)

    # Print action if provided
    if action:
        action_names = {
            Direction.UP: "UP",
            Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT",
            Direction.RIGHT: "RIGHT",
        }
        print(f"\n{action_names[action]}")
    print()


def run_training(args):
    """Run training sessions

    Args:
        args: Command line arguments
    """
    # Create environment and agent
    env = Environment(board_size=args.board_size)
    agent = QLearningAgent(
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    # Load model if specified
    if args.load:
        # Auto-prepend models/ if just filename given
        load_path = args.load
        if not os.path.dirname(load_path):  # No directory in path
            load_path = os.path.join("models", load_path)

        if os.path.exists(load_path):
            print(f"Load trained model from {load_path}")
            agent.load_model(load_path)
        else:
            print(f"Warning: Model file {load_path} not found, starting fresh")

    # Set learning mode
    if args.dontlearn:
        agent.set_learning_enabled(False)
        print("Learning disabled - pure exploitation mode")
    else:
        # Print hyperparameters
        print(f"\n{'=' * 60}")
        print("Q-Learning Hyperparameters:")
        print(f"{'=' * 60}")
        print(f"  Learning Rate (α): {args.lr}")
        print(f"  Discount Factor (γ): {args.gamma}")
        print(f"  Epsilon Start: {args.epsilon}")
        print(f"  Epsilon Decay: {args.epsilon_decay}")
        print(f"  Epsilon Min: {args.epsilon_min}")
        print(f"{'=' * 60}\n")

    # Create display if visual mode
    display = None
    if args.visual:
        if not DISPLAY_AVAILABLE:
            print("Warning: pygame not available, running in headless mode")
            args.visual = False
        else:
            display = Display(board_size=args.board_size)

    # Determine FPS
    fps = 0
    if args.visual and not args.step_by_step:
        if args.speed == "slow":
            fps = 2
        elif args.speed == "medium":
            fps = 10
        elif args.speed == "fast":
            fps = 30
        elif args.speed == "ultrafast":
            fps = 60

    # Training loop
    all_max_lengths = []
    all_durations = []

    # Max steps to prevent infinite loops
    MAX_STEPS = 1000

    for session in range(1, args.sessions + 1):
        state = env.reset()
        done = False
        session_reward = 0
        steps = 0

        while not done and steps < MAX_STEPS:
            steps += 1
            # Compute safe actions (don't move into immediate death)
            safe_actions = get_safe_actions(env, state)

            # Get action from agent (only from safe actions)
            action = agent.get_action(state, safe_actions)

            # Show state in terminal if verbose mode
            if args.verbose:
                print_vision(state, action)

            # Take step in environment
            next_state, reward, done = env.step(action)
            session_reward += reward

            # Update agent
            agent.update(state, action, reward, next_state, done)

            state = next_state

            # Display if visual mode
            if display:
                board_state = env.get_board_state()
                display.draw(board_state, session, args.sessions)
                if args.step_by_step:
                    display.wait_for_step()
                else:
                    display.set_fps(fps)

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Track statistics
        all_max_lengths.append(env.max_length)
        all_durations.append(env.steps)

        # Print session summary
        # if args.verbose or session == args.sessions or session % 100 == 0:
        reward_colored = colorize_reward(session_reward)
        print(
            f"Session {session}/{args.sessions}: "
            f"Length={env.max_length}, "
            f"Steps={env.steps}, "
            f"Reward={reward_colored}, "
            f"Epsilon={agent.epsilon:.3f}"
        )

    # Final statistics
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Total sessions: {args.sessions}")
    print(
        f"Average max length: {
            sum(all_max_lengths) /
            len(all_max_lengths):.2f}")
    print(f"Best max length: {max(all_max_lengths)}")
    print(f"Average duration: {sum(all_durations) / len(all_durations):.2f}")
    print(f"States explored: {agent.get_stats()['states_explored']}")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    # Bonus achievements
    print(f"\n{'=' * 60}")
    print("BONUS ACHIEVEMENTS:")
    print(f"{'=' * 60}")
    best = max(all_max_lengths)
    bonus_levels = [
        (35, "LEGENDARY"),
        (30, "MASTER"),
        (25, "EXPERT"),
        (20, "ADVANCED"),
        (15, "PROFICIENT"),
        (10, "COMPETENT"),
    ]
    for level, rank in bonus_levels:
        if best >= level:
            print(f"✓ Length {level}+ achieved! Rank: {rank}")
            break
    else:
        print(f"  Current best: {best} (Goal: 10+)")

    if args.load:
        print(
            f"✓ Model transfer: Trained model works on {
                args.board_size}x{
                args.board_size} board")
    print(f"{'=' * 60}")

    # Save model if specified
    if args.save:
        # Auto-prepend models/ if just filename given
        save_path = args.save
        if not os.path.dirname(save_path):  # No directory in path
            save_path = os.path.join("models", save_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save_model(save_path)
        print(f"\nSave learning state in {save_path}")

    # Close display
    if display:
        display.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Learn2Slither - Snake with Q-Learning"
    )

    # Required/Common arguments
    parser.add_argument(
        "-sessions",
        type=int,
        default=1,
        help="Number of training sessions to run (default: 1)",
    )

    # Model arguments
    parser.add_argument(
        "-save",
        type=str,
        help="Path to save the trained model")
    parser.add_argument(
        "-load",
        type=str,
        help="Path to load a pre-trained model")

    # Display arguments
    parser.add_argument(
        "-visual",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable/disable visual display (default: on)",
    )
    parser.add_argument(
        "-speed",
        type=str,
        choices=["slow", "medium", "fast", "ultrafast"],
        default="medium",
        help="Display speed (default: medium)",
    )
    parser.add_argument(
        "-step-by-step",
        action="store_true",
        help="Enable step-by-step mode (press space to advance)",
    )

    # Learning arguments
    parser.add_argument(
        "-dontlearn",
        action="store_true",
        help="Disable learning (evaluation mode)")

    # Q-Learning hyperparameters
    parser.add_argument(
        "-lr",
        type=float,
        default=0.2,
        help="Learning rate (alpha) (default: 0.2)")
    parser.add_argument(
        "-gamma",
        type=float,
        default=0.9,
        help="Discount factor (gamma) (default: 0.9)")
    parser.add_argument(
        "-epsilon",
        type=float,
        default=1.0,
        help="Initial exploration rate (default: 1.0)",
    )
    parser.add_argument(
        "-epsilon_decay",
        type=float,
        default=0.999,
        help="Epsilon decay rate (default: 0.999)",
    )
    parser.add_argument(
        "-epsilon_min",
        type=float,
        default=0.05,
        help="Minimum epsilon value (default: 0.05)",
    )

    # Board arguments
    parser.add_argument(
        "-board_size",
        type=int,
        default=10,
        help="Size of the board (default: 10)")

    # Verbose output
    parser.add_argument(
        "-verbose",
        action="store_true",
        help="Print state/vision during training")

    args = parser.parse_args()

    # Convert visual string to boolean
    args.visual = args.visual == "on"

    # Run training
    try:
        run_training(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    # If no arguments provided, launch graphical menu
    if len(sys.argv) == 1:
        try:
            gui.main()
        except ImportError:
            print("GUI menu module not found. Using command-line mode.")
            print("Run with --help for usage information.")
            main()
    else:
        main()

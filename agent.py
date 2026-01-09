from environment import Direction
import pickle
import random


class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        eval_epsilon=0.0,
    ):
        """Initialize Q-Learning agent

        Args:
            learning_rate: Learning rate (alpha) - higher for faster learning
            discount_factor: Discount factor (gamma) - lower for
                immediate rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays - slower decay
            epsilon_min: Minimum epsilon value - keep some exploration
            eval_epsilon: Epsilon for evaluation (0.0 = pure exploitation)
        """
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eval_epsilon = eval_epsilon
        self.learning_enabled = True

    def _state_to_key(self, state):
        """Convert state dictionary to hashable key

        Args:
            state: State dictionary from environment

        Returns:
            tuple: Hashable state representation using simplified features
        """
        # Use simplified features for much smaller state space
        return state["features"]

    def get_action(self, state, safe_actions=None):
        """Get action using epsilon-greedy policy with safety

        Args:
            state: Current state from environment
            safe_actions: List of safe actions (or None for all actions)

        Returns:
            Direction: Chosen action
        """
        state_key = self._state_to_key(state)

        # Get available actions (only safe ones if provided)
        available_actions = safe_actions if safe_actions else list(Direction)

        # If no safe actions, choose any (trapped situation)
        if not available_actions:
            available_actions = list(Direction)

        # Exploration: random action from available actions
        # Use eval_epsilon when learning disabled to prevent deterministic
        # loops
        current_epsilon = (
            self.epsilon if self.learning_enabled else self.eval_epsilon
        )
        if random.random() < current_epsilon:
            return random.choice(available_actions)

        # Exploitation: best action from Q-table among available actions
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in Direction}

        # Get action with highest Q-value among safe actions
        q_values = self.q_table[state_key]
        available_q = {action: q_values[action]
                       for action in available_actions}

        max_q = max(available_q.values())
        best_actions = [
            action for action,
            q in available_q.items()
            if q == max_q
        ]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Update Q-table based on experience

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
            done: Whether episode is done
        """
        if not self.learning_enabled:
            return

        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Initialize Q-values if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in Direction}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in Direction}

        # Get current Q-value
        current_q = self.q_table[state_key][action]

        # Calculate max Q-value for next state
        if done:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state_key].values())

        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate (only when learning is enabled)"""
        if self.learning_enabled and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        """Save Q-table to file

        Args:
            filepath: Path to save file
        """
        model_data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "eval_epsilon": self.eval_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """Load Q-table from file

        Args:
            filepath: Path to model file
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.q_table = model_data["q_table"]
        self.epsilon = model_data.get("epsilon", self.epsilon)
        self.learning_rate = model_data.get(
            "learning_rate", self.learning_rate)
        self.discount_factor = model_data.get(
            "discount_factor", self.discount_factor)
        self.eval_epsilon = model_data.get("eval_epsilon", self.eval_epsilon)
        self.epsilon_decay = model_data.get(
            "epsilon_decay", self.epsilon_decay)
        self.epsilon_min = model_data.get("epsilon_min", self.epsilon_min)

    def set_learning_enabled(self, enabled):
        """Enable or disable learning

        Args:
            enabled: Whether learning should be enabled
        """
        self.learning_enabled = enabled

    def get_stats(self):
        """Get agent statistics

        Returns:
            dict: Statistics about the agent
        """
        return {
            "states_explored": len(self.q_table),
            "epsilon": self.epsilon,
            "learning_enabled": self.learning_enabled,
        }

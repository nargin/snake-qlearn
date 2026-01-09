import random
from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class CellType(Enum):
    EMPTY = '0'
    WALL = 'W'
    SNAKE_HEAD = 'H'
    SNAKE_BODY = 'S'
    GREEN_APPLE = 'G'
    RED_APPLE = 'R'


class Environment:
    def __init__(self, board_size=10):
        """Initialize the environment

        Args:
            board_size: Size of the board (default 10x10)
        """
        self.board_size = board_size
        self.snake = []
        self.direction = Direction.RIGHT
        self.green_apples = []
        self.red_apple = None
        self.game_over = False
        self.steps = 0
        self.max_length = 3

        # Reward constants
        self.DEATH_REWARD = -500
        self.REVISIT_PENALTY = -10
        self.MOVE_PENALTY = -0.1
        self.GREEN_APPLE_REWARD = 100
        self.RED_APPLE_PENALTY = -110
        self.MOVE_CLOSER_BONUS = 2
        self.MOVE_AWAY_PENALTY = -5

        # Track visited positions to prevent loops
        self.visited_positions = set()

        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.game_over = False
        self.steps = 0

        self._init_snake()

        # Place apples
        self.green_apples = []
        self._place_apple('green')
        self._place_apple('green')
        self._place_apple('red')

        self.max_length = len(self.snake)

        # Clear visited positions for new episode
        self.visited_positions = set()
        # Add initial snake head position
        self.visited_positions.add(self.snake[0])

        return self.get_state()

    def _init_snake(self):
        """Initialize snake at random position"""
        # Choose random starting position with enough space
        while True:
            start_x = random.randint(3, self.board_size - 4)
            start_y = random.randint(3, self.board_size - 4)

            # Choose random horizontal or vertical orientation
            if random.choice([True, False]):
                # Horizontal
                self.snake = [
                    (start_x, start_y),
                    (start_x - 1, start_y),
                    (start_x - 2, start_y)
                ]
                self.direction = Direction.RIGHT
            else:
                # Vertical
                self.snake = [
                    (start_x, start_y),
                    (start_x, start_y - 1),
                    (start_x, start_y - 2)
                ]
                self.direction = Direction.DOWN
            break

    def _place_apple(self, apple_type):
        """Place an apple at random empty position

        Args:
            apple_type: 'green' or 'red'
        """
        while True:
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            pos = (x, y)

            # Check if position is empty
            if (pos not in self.snake and
                pos not in self.green_apples and
                pos != self.red_apple):
                if apple_type == 'green':
                    self.green_apples.append(pos)
                else:
                    self.red_apple = pos
                break

    def get_state(self):
        """Get current state (snake's vision in 4 directions)

        Returns:
            Dictionary with vision in each direction and simplified features
        """
        head_x, head_y = self.snake[0]

        # Get vision in all 4 directions (for display)
        vision = {
            'up': self._get_vision_line(head_x, head_y, 0, -1),
            'down': self._get_vision_line(head_x, head_y, 0, 1),
            'left': self._get_vision_line(head_x, head_y, -1, 0),
            'right': self._get_vision_line(head_x, head_y, 1, 0)
        }

        # Simplified features for learning
        vision['features'] = self._get_simplified_features()

        return vision

    def _get_simplified_features(self):
        """Get simplified state features for fast Q-learning

        Uses binary danger detection (1 step ahead) and compass apple direction.
        Creates small state space (~512 states) for rapid learning.

        Returns:
            tuple: State features (danger_up, danger_right, danger_down, danger_left,
                   apple_direction, current_direction)
        """
        head_x, head_y = self.snake[0]
        features = []

        # Binary danger detection (1 step ahead in 4 directions)
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, Right, Down, Left
            next_x = head_x + dx
            next_y = head_y + dy

            # Check if this direction leads to immediate danger
            is_danger = (
                next_x < 0 or next_x >= self.board_size or
                next_y < 0 or next_y >= self.board_size or
                (next_x, next_y) in self.snake
            )
            features.append(1 if is_danger else 0)

        # Apple direction in 8 compass directions (N, NE, E, SE, S, SW, W, NW)
        apple_compass = self._get_apple_compass_direction(self.green_apples)
        features.append(apple_compass)

        # Current direction
        features.append(self.direction.value)

        return tuple(features)

    def _bucket_distance(self, distance):
        """Bucket distances to reduce state space

        Args:
            distance: Raw distance value

        Returns:
            int: Bucketed distance (0=immediate, 1=near, 2=mid, 3=far)
        """
        if distance == 1:
            return 0  # Immediate (next to it)
        elif distance <= 3:
            return 1  # Near
        elif distance <= 6:
            return 2  # Mid
        else:
            return 3  # Far

    def _get_distance_to_obstacle(self, start_x, start_y, dx, dy):
        """Get distance to first obstacle in a direction

        Args:
            start_x, start_y: Starting position
            dx, dy: Direction delta

        Returns:
            int: Distance to obstacle (wall or snake body)
        """
        distance = 0
        x, y = start_x + dx, start_y + dy

        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            distance += 1
            pos = (x, y)

            # Check if snake body
            if pos in self.snake:
                return distance

            x += dx
            y += dy

        # Hit wall
        return distance + 1

    def _get_distance_to_item(self, start_x, start_y, dx, dy, items):
        """Get distance to nearest item in a direction

        Args:
            start_x, start_y: Starting position
            dx, dy: Direction delta
            items: List of item positions

        Returns:
            int: Distance to item, or board_size if not visible
        """
        if not items:
            return self.board_size

        distance = 0
        x, y = start_x + dx, start_y + dy

        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            distance += 1
            pos = (x, y)

            if pos in items:
                return distance

            x += dx
            y += dy

        return self.board_size

    def _get_apple_compass_direction(self, apples):
        """Get compass direction to nearest apple (8 directions)

        Args:
            apples: List of apple positions

        Returns:
            int: 0-7 representing N, NE, E, SE, S, SW, W, NW (or 8 if no apple)
        """
        if not apples:
            return 8  # No apple visible

        head_x, head_y = self.snake[0]

        # Find nearest apple
        min_dist = float('inf')
        nearest = None
        for apple in apples:
            dist = abs(apple[0] - head_x) + abs(apple[1] - head_y)
            if dist < min_dist:
                min_dist = dist
                nearest = apple

        if not nearest:
            return 8

        # Calculate relative position
        dx = nearest[0] - head_x
        dy = nearest[1] - head_y

        # Convert to 8 compass directions
        if dy < 0:  # North
            if dx > 0:
                return 1  # NE
            elif dx < 0:
                return 7  # NW
            else:
                return 0  # N
        elif dy > 0:  # South
            if dx > 0:
                return 3  # SE
            elif dx < 0:
                return 5  # SW
            else:
                return 4  # S
        else:  # Same horizontal
            if dx > 0:
                return 2  # E
            else:
                return 6  # W

    def _get_apple_direction(self, apples):
        """Get direction indicators to nearest apple

        Args:
            apples: List of apple positions

        Returns:
            list: [up, right, down, left] - 1 if apple in that direction, 0 otherwise
        """
        if not apples:
            return [0, 0, 0, 0]

        head_x, head_y = self.snake[0]

        # Find nearest apple
        min_dist = float('inf')
        nearest = None
        for apple in apples:
            dist = abs(apple[0] - head_x) + abs(apple[1] - head_y)
            if dist < min_dist:
                min_dist = dist
                nearest = apple

        if not nearest:
            return [0, 0, 0, 0]

        # Encode direction to nearest apple
        dx = nearest[0] - head_x
        dy = nearest[1] - head_y

        direction = [0, 0, 0, 0]  # up, right, down, left
        if dy < 0:  # Apple is above
            direction[0] = 1
        if dx > 0:  # Apple is to the right
            direction[1] = 1
        if dy > 0:  # Apple is below
            direction[2] = 1
        if dx < 0:  # Apple is to the left
            direction[3] = 1

        return direction

    def _get_vision_line(self, start_x, start_y, dx, dy):
        """Get vision in one direction from a position

        Args:
            start_x, start_y: Starting position
            dx, dy: Direction delta

        Returns:
            String representing vision in that direction
        """
        vision = []
        x, y = start_x + dx, start_y + dy

        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            pos = (x, y)

            if pos in self.green_apples:
                vision.append(CellType.GREEN_APPLE.value)
            elif pos == self.red_apple:
                vision.append(CellType.RED_APPLE.value)
            elif pos == self.snake[0]:
                vision.append(CellType.SNAKE_HEAD.value)
            elif pos in self.snake[1:]:
                vision.append(CellType.SNAKE_BODY.value)
            else:
                vision.append(CellType.EMPTY.value)

            x += dx
            y += dy

        # Add wall at the end
        vision.append(CellType.WALL.value)

        return ''.join(vision)

    def step(self, action):
        """Take a step in the environment

        Args:
            action: Direction to move (Direction enum)

        Returns:
            tuple: (state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0, True

        self.steps += 1

        # Store old position for reward shaping
        old_head = self.snake[0]
        old_distance = self._get_min_distance_to_apples(
            old_head, self.green_apples
        )

        # Update direction (prevent 180-degree turns)
        if not self._is_opposite_direction(action):
            self.direction = action

        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self._get_direction_delta(self.direction)
        new_head = (head_x + dx, head_y + dy)

        # Check for wall collision
        if (new_head[0] < 0 or new_head[0] >= self.board_size or
            new_head[1] < 0 or new_head[1] >= self.board_size):
            self.game_over = True
            return self.get_state(), self.DEATH_REWARD, True

        # Self collision
        if new_head in self.snake:
            self.game_over = True
            return self.get_state(), self.DEATH_REWARD, True

        # Move snake
        self.snake.insert(0, new_head)
        reward = self.MOVE_PENALTY

        # Penalize revisiting positions to prevent loops
        if new_head in self.visited_positions:
            reward += self.REVISIT_PENALTY

        self.visited_positions.add(new_head)

        # Reward shaping: stronger guidance towards green apples
        new_distance = self._get_min_distance_to_apples(
            new_head, self.green_apples
        )
        if new_distance < old_distance:
            reward += self.MOVE_CLOSER_BONUS
        elif new_distance > old_distance:
            reward += self.MOVE_AWAY_PENALTY

        # Check for green apple
        if new_head in self.green_apples:
            self.green_apples.remove(new_head)
            self._place_apple('green')
            reward = self.GREEN_APPLE_REWARD
            if len(self.snake) > self.max_length:
                self.max_length = len(self.snake)
        # Check for red apple
        elif new_head == self.red_apple:
            self.red_apple = None
            self._place_apple('red')
            reward = self.RED_APPLE_PENALTY
            # Remove tail twice (once for move, once for penalty)
            self.snake.pop()
            if len(self.snake) > 1:
                self.snake.pop()
            else:
                # Length dropped to 0
                self.game_over = True
                return self.get_state(), self.DEATH_REWARD, True
        else:
            # No apple, remove tail normally
            self.snake.pop()

        return self.get_state(), reward, False

    def _is_opposite_direction(self, new_direction):
        """Check if new direction is opposite to current direction

        Args:
            new_direction: Direction to check

        Returns:
            bool: True if opposite direction
        """
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return opposites[self.direction] == new_direction

    def _get_direction_delta(self, direction):
        """Get x, y delta for a direction

        Args:
            direction: Direction enum

        Returns:
            tuple: (dx, dy)
        """
        deltas = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0)
        }
        return deltas[direction]

    def _get_min_distance_to_apples(self, position, apples):
        """Get Manhattan distance to nearest apple

        Args:
            position: Current position (x, y)
            apples: List of apple positions

        Returns:
            int: Distance to nearest apple (or board_size*2 if no apples)
        """
        if not apples:
            return self.board_size * 2

        min_dist = float('inf')
        for apple in apples:
            dist = abs(apple[0] - position[0]) + abs(apple[1] - position[1])
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def get_board_state(self):
        """Get full board state for visualization

        Returns:
            dict: Contains snake, apples, and board info
        """
        return {
            'snake': self.snake.copy(),
            'green_apples': self.green_apples.copy(),
            'red_apple': self.red_apple,
            'board_size': self.board_size,
            'length': len(self.snake),
            'max_length': self.max_length,
            'steps': self.steps
        }

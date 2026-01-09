import pygame
import sys


class Display:
    def __init__(self, board_size=10, cell_size=40):
        """Initialize display

        Args:
            board_size: Size of the board
            cell_size: Size of each cell in pixels
        """
        self.board_size = board_size
        self.cell_size = cell_size
        self.width = board_size * cell_size
        self.height = board_size * cell_size + 100  # Extra space for stats

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Learn2Slither - Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (34, 139, 34)
        self.LIGHT_GREEN = (144, 238, 144)
        self.RED = (220, 20, 60)
        self.DARK_RED = (139, 0, 0)
        self.BLUE = (30, 144, 255)
        self.DARK_BLUE = (0, 100, 200)
        self.LIGHT_BLUE = (135, 206, 250)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (220, 220, 220)
        self.BG_COLOR = (240, 248, 255)  # Light blue background

    def draw(self, board_state, session_num=0, total_sessions=0):
        """Draw the current board state

        Args:
            board_state: Dictionary containing board information
            session_num: Current session number
            total_sessions: Total number of sessions
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.BG_COLOR)

        # Draw checkered background
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x + y) % 2 == 0:
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self.screen, (250, 250, 255), rect)

        # Draw grid lines
        for x in range(self.board_size + 1):
            pygame.draw.line(
                self.screen,
                self.LIGHT_GRAY,
                (x * self.cell_size, 0),
                (x * self.cell_size, self.board_size * self.cell_size),
            )
        for y in range(self.board_size + 1):
            pygame.draw.line(
                self.screen,
                self.LIGHT_GRAY,
                (0, y * self.cell_size),
                (self.board_size * self.cell_size, y * self.cell_size),
            )

        # Draw green apples
        for apple in board_state["green_apples"]:
            self._draw_circle(apple[0], apple[1], self.GREEN)

        # Draw red apple
        if board_state["red_apple"]:
            self._draw_circle(
                board_state["red_apple"][0],
                board_state["red_apple"][1],
                self.RED)

        # Draw snake
        snake = board_state["snake"]
        if snake:
            # Draw head
            self._draw_rect(snake[0][0], snake[0][1], self.DARK_BLUE)
            # Draw body
            for segment in snake[1:]:
                self._draw_rect(segment[0], segment[1], self.BLUE)

        # Draw stats
        self._draw_stats(board_state, session_num, total_sessions)

        pygame.display.flip()

    def _draw_rect(self, x, y, color):
        """Draw a filled rectangle at grid position with gradient effect

        Args:
            x, y: Grid coordinates
            color: Color tuple
        """
        base_x = x * self.cell_size + 2
        base_y = y * self.cell_size + 2
        size = self.cell_size - 4

        # Main rectangle
        rect = pygame.Rect(base_x, base_y, size, size)
        pygame.draw.rect(self.screen, color, rect)

        # Add highlight effect (lighter top-left corner)
        highlight_color = tuple(min(c + 40, 255) for c in color)
        highlight_rect = pygame.Rect(base_x, base_y, size // 2, size // 2)
        pygame.draw.rect(self.screen, highlight_color, highlight_rect)

        # Add border for depth
        border_color = tuple(max(c - 30, 0) for c in color)
        pygame.draw.rect(self.screen, border_color, rect, 2)

    def _draw_circle(self, x, y, color):
        """Draw a filled circle at grid position with shine effect

        Args:
            x, y: Grid coordinates
            color: Color tuple
        """
        center = (
            x * self.cell_size + self.cell_size // 2,
            y * self.cell_size + self.cell_size // 2,
        )
        radius = self.cell_size // 3

        # Draw main circle
        pygame.draw.circle(self.screen, color, center, radius)

        # Add darker border for depth
        border_color = tuple(max(c - 50, 0) for c in color)
        pygame.draw.circle(self.screen, border_color, center, radius, 2)

        # Add shine spot (lighter circle in top-left)
        shine_color = tuple(min(c + 80, 255) for c in color)
        shine_center = (center[0] - radius // 3, center[1] - radius // 3)
        shine_radius = radius // 3
        pygame.draw.circle(
            self.screen,
            shine_color,
            shine_center,
            shine_radius)

    def _draw_stats(self, board_state, session_num, total_sessions):
        """Draw statistics below the board

        Args:
            board_state: Board state dictionary
            session_num: Current session number
            total_sessions: Total sessions
        """
        y_offset = self.board_size * self.cell_size + 10

        # Session info
        session_text = f"Session: {session_num}/{total_sessions}"
        text_surface = self.font.render(session_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, y_offset))

        # Length info
        length_text = f"Length: {board_state['length']}"
        text_surface = self.font.render(length_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, y_offset + 25))

        # Max length
        max_length_text = f"Max Length: {board_state['max_length']}"
        text_surface = self.font.render(max_length_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, y_offset + 50))

        # Steps
        steps_text = f"Steps: {board_state['steps']}"
        text_surface = self.font.render(steps_text, True, self.BLACK)
        self.screen.blit(text_surface, (200, y_offset + 25))

    def set_fps(self, fps):
        """Set the frames per second

        Args:
            fps: Frames per second
        """
        if fps > 0:
            self.clock.tick(fps)

    def wait_for_step(self):
        """Wait for user to press a key (step-by-step mode)"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if (event.key == pygame.K_SPACE
                            or event.key == pygame.K_RETURN):
                        waiting = False

    def close(self):
        """Close the display"""
        pygame.quit()

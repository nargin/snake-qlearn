import pygame
import os
import subprocess


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
BLUE = (70, 130, 180)
DARK_BLUE = (25, 25, 112)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
HOVER_COLOR = (100, 149, 237)


class Button:
    """Simple button class"""

    def __init__(
            self,
            x,
            y,
            width,
            height,
            text,
            color=BLUE,
            hover_color=HOVER_COLOR):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=10)

        text_surface = font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def update(self, mouse_pos):
        """Update hover state"""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos):
        """Check if button was clicked"""
        return self.rect.collidepoint(mouse_pos)


class InputBox:
    """Text input box"""

    def __init__(self, x, y, width, height, default_text=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = GRAY
        self.text = default_text
        self.active = False

    def draw(self, screen, font):
        color = BLUE if self.active else GRAY
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=5)

        text_surface = font.render(self.text, True, BLACK)
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))

    def handle_click(self, mouse_pos):
        """Handle mouse click"""
        self.active = self.rect.collidepoint(mouse_pos)

    def handle_key(self, event):
        """Handle keyboard input"""
        if not self.active:
            return False

        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.key == pygame.K_RETURN:
            return True
        elif len(self.text) < 20:
            # Allow digits, decimal point, and letters (for model names)
            if event.unicode.isprintable():
                self.text += event.unicode
        return False


class Menu:
    """Main menu class"""

    def __init__(self):
        self.init_pygame()
        self.running = True
        self.current_screen = "main"

    def init_pygame(self):
        """Initialize or reinitialize pygame"""
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Learn2Slither - Menu")
        self.clock = pygame.time.Clock()

        # Fonts
        self.title_font = pygame.font.Font(None, 64)
        self.button_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def draw_background(self):
        """Draw checkered background"""
        self.screen.fill(LIGHT_BLUE)

        # Checkered pattern
        cell_size = 40
        for x in range(0, self.width, cell_size):
            for y in range(0, self.height, cell_size):
                if (x // cell_size + y // cell_size) % 2 == 0:
                    rect = pygame.Rect(x, y, cell_size, cell_size)
                    pygame.draw.rect(self.screen, (183, 226, 240), rect)

    def draw_title(self, text, y=50):
        """Draw centered title"""
        title = self.title_font.render(text, True, DARK_BLUE)
        title_rect = title.get_rect(center=(self.width // 2, y))

        # Shadow
        shadow = self.title_font.render(text, True, BLACK)
        shadow_rect = shadow.get_rect(center=(self.width // 2 + 2, y + 2))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(title, title_rect)

    def main_menu(self):
        """Main menu screen"""
        buttons = [
            Button(250, 200, 300, 80, "Train", color=GREEN),
            Button(250, 300, 300, 80, "Select Model", color=BLUE),
            Button(250, 400, 300, 80, "Exit", color=RED),
        ]

        while self.running and self.current_screen == "main":
            mouse_pos = pygame.mouse.get_pos()

            self.draw_background()
            self.draw_title("LEARN2SLITHER")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, button in enumerate(buttons):
                        if button.is_clicked(mouse_pos):
                            if i == 0:
                                self.train_menu()
                            elif i == 1:
                                self.select_model_menu()
                            elif i == 2:
                                self.running = False
                                return

            # Update hover states
            for button in buttons:
                button.update(mouse_pos)
                button.draw(self.screen, self.button_font)

            pygame.display.flip()
            self.clock.tick(60)

    def train_menu(self):
        """Training configuration screen"""
        self.current_screen = "train"

        # Basic inputs
        sessions_input = InputBox(250, 120, 150, 35, "1000")
        name_input = InputBox(250, 165, 150, 35, "model")

        # Parameter inputs (with defaults)
        lr_input = InputBox(550, 120, 150, 35, "0.2")
        gamma_input = InputBox(550, 165, 150, 35, "0.9")
        epsilon_input = InputBox(550, 210, 150, 35, "1.0")
        decay_input = InputBox(550, 255, 150, 35, "0.999")
        min_eps_input = InputBox(550, 300, 150, 35, "0.05")

        all_inputs = [
            sessions_input,
            name_input,
            lr_input,
            gamma_input,
            epsilon_input,
            decay_input,
            min_eps_input,
        ]

        start_button = Button(200, 460, 400, 60, "Start Training", color=GREEN)
        back_button = Button(200, 530, 400, 60, "Back", color=GRAY)

        while self.running and self.current_screen == "train":
            mouse_pos = pygame.mouse.get_pos()

            self.draw_background()
            self.draw_title("TRAIN MODEL", 30)

            # Left column labels (basic)
            label1 = self.small_font.render("Sessions:", True, BLACK)
            label2 = self.small_font.render("Model Name:", True, BLACK)
            self.screen.blit(label1, (100, 125))
            self.screen.blit(label2, (100, 170))

            # Right column labels (parameters)
            label3 = self.small_font.render("Learning Rate:", True, BLACK)
            label4 = self.small_font.render("Discount (γ):", True, BLACK)
            label5 = self.small_font.render("Epsilon Start:", True, BLACK)
            label6 = self.small_font.render("Epsilon Decay:", True, BLACK)
            label7 = self.small_font.render("Epsilon Min:", True, BLACK)
            self.screen.blit(label3, (420, 125))
            self.screen.blit(label4, (420, 170))
            self.screen.blit(label5, (420, 215))
            self.screen.blit(label6, (420, 260))
            self.screen.blit(label7, (420, 305))

            # Info text
            info = self.small_font.render(
                "Leave default for standard Q-learning", True, DARK_GRAY
            )
            self.screen.blit(info, (200, 360))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for inp in all_inputs:
                        inp.handle_click(mouse_pos)

                    if start_button.is_clicked(mouse_pos):
                        sessions = sessions_input.text or "1000"
                        name = name_input.text or "model"

                        # Get parameters
                        params = {
                            "lr": lr_input.text or "0.2",
                            "gamma": gamma_input.text or "0.9",
                            "epsilon": epsilon_input.text or "1.0",
                            "decay": decay_input.text or "0.999",
                            "min_eps": min_eps_input.text or "0.05",
                        }

                        self.run_training(sessions, name, params)
                        self.current_screen = "main"
                        return

                    if back_button.is_clicked(mouse_pos):
                        self.current_screen = "main"
                        return

                if event.type == pygame.KEYDOWN:
                    for inp in all_inputs:
                        inp.handle_key(event)

            # Update hover states
            start_button.update(mouse_pos)
            back_button.update(mouse_pos)

            # Draw everything
            for inp in all_inputs:
                inp.draw(self.screen, self.small_font)

            start_button.draw(self.screen, self.button_font)
            back_button.draw(self.screen, self.button_font)

            pygame.display.flip()
            self.clock.tick(60)

    def select_model_menu(self):
        """Model selection screen with pagination"""
        self.current_screen = "select_model"

        models = self.list_models()
        if not models:
            self.show_message("No models found!")
            self.current_screen = "main"
            return

        page = 0
        models_per_page = 5

        while self.running and self.current_screen == "select_model":
            mouse_pos = pygame.mouse.get_pos()

            # Calculate pagination
            total_pages = (
                len(models) + models_per_page - 1) // models_per_page
            start_idx = page * models_per_page
            end_idx = min(start_idx + models_per_page, len(models))
            current_models = models[start_idx:end_idx]

            # Create buttons for current page models
            buttons = []
            y = 150
            for i, model in enumerate(current_models):
                buttons.append(Button(150, y, 500, 50, model, color=DARK_BLUE))
                y += 60

            # Navigation buttons
            prev_button = Button(150, 480, 100, 50, "< Prev", color=GRAY)
            next_button = Button(550, 480, 100, 50, "Next >", color=GRAY)
            back_button = Button(300, 540, 200, 50, "Back", color=RED)

            self.draw_background()
            self.draw_title("SELECT MODEL", 40)

            # Show page info
            page_text = self.small_font.render(
                f"Page {page + 1} / {total_pages}", True, BLACK
            )
            self.screen.blit(page_text, (340, 100))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check model buttons
                    for i, button in enumerate(buttons):
                        if button.is_clicked(mouse_pos):
                            selected_model = models[start_idx + i]
                            self.model_action_menu(selected_model)
                            if self.current_screen == "main":
                                return

                    # Navigation
                    if prev_button.is_clicked(mouse_pos) and page > 0:
                        page -= 1
                    if next_button.is_clicked(
                            mouse_pos) and page < total_pages - 1:
                        page += 1
                    if back_button.is_clicked(mouse_pos):
                        self.current_screen = "main"
                        return

            # Update and draw buttons
            for button in buttons:
                button.update(mouse_pos)
                button.draw(self.screen, self.small_font)

            # Draw navigation
            if page > 0:
                prev_button.update(mouse_pos)
                prev_button.draw(self.screen, self.small_font)

            if page < total_pages - 1:
                next_button.update(mouse_pos)
                next_button.draw(self.screen, self.small_font)

            back_button.update(mouse_pos)
            back_button.draw(self.screen, self.small_font)

            pygame.display.flip()
            self.clock.tick(60)

    def model_action_menu(self, model_name):
        """Choose action for selected model"""
        self.current_screen = "model_action"

        eval_button = Button(250, 200, 300, 80, "Evaluate", color=BLUE)
        watch_button = Button(250, 300, 300, 80, "Watch Play", color=GREEN)
        back_button = Button(250, 400, 300, 80, "Back", color=GRAY)

        while self.running and self.current_screen == "model_action":
            mouse_pos = pygame.mouse.get_pos()

            self.draw_background()
            self.draw_title(model_name[:20], 40)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if eval_button.is_clicked(mouse_pos):
                        self.run_evaluation(model_name)
                        self.current_screen = "main"
                        return
                    elif watch_button.is_clicked(mouse_pos):
                        self.run_watch(model_name)
                        self.current_screen = "main"
                        return
                    elif back_button.is_clicked(mouse_pos):
                        self.current_screen = "select_model"
                        return

            eval_button.update(mouse_pos)
            watch_button.update(mouse_pos)
            back_button.update(mouse_pos)

            eval_button.draw(self.screen, self.button_font)
            watch_button.draw(self.screen, self.button_font)
            back_button.draw(self.screen, self.button_font)

            pygame.display.flip()
            self.clock.tick(60)

    def show_message(self, message):
        """Show a message screen"""
        start_time = pygame.time.get_ticks()

        while pygame.time.get_ticks() - start_time < 1500:
            self.draw_background()

            text = self.button_font.render(message, True, BLACK)
            text_rect = text.get_rect(
                center=(
                    self.width // 2,
                    self.height // 2))
            self.screen.blit(text, text_rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

            pygame.display.flip()
            self.clock.tick(60)

    def list_models(self):
        """List available model files"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            return []

        models = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        return sorted(models)

    def run_training(self, sessions, name, params=None):
        """Run training in subprocess with optional parameters"""
        pygame.quit()
        cmd = [
            "python3",
            "snake.py",
            "-sessions",
            sessions,
            "-save",
            f"{name}.pkl",
            "-visual",
            "off",
        ]

        # Add optional parameters if provided
        if params:
            cmd.extend(["-lr", params["lr"]])
            cmd.extend(["-gamma", params["gamma"]])
            cmd.extend(["-epsilon", params["epsilon"]])
            cmd.extend(["-epsilon_decay", params["decay"]])
            cmd.extend(["-epsilon_min", params["min_eps"]])

        try:
            subprocess.run(cmd)
        except Exception as e:
            print(f"Error running training: {e}")
        finally:
            # Reinitialize pygame
            self.init_pygame()

    def run_evaluation(self, model_name):
        """Run model evaluation"""
        pygame.quit()
        cmd = [
            "python3",
            "snake.py",
            "-load",
            model_name,
            "-sessions",
            "5",
            "-dontlearn",
            "-visual",
            "off",
        ]
        try:
            subprocess.run(cmd)
        except Exception as e:
            print(f"Error running evaluation: {e}")
        finally:
            # Reinitialize pygame
            self.init_pygame()

    def run_watch(self, model_name):
        """Watch model play with visualization"""
        pygame.quit()
        cmd = [
            "python3",
            "snake.py",
            "-load",
            model_name,
            "-sessions",
            "3",
            "-dontlearn",
            "-visual",
            "on",
            "-speed",
            "medium",
        ]
        try:
            subprocess.run(cmd)
        except Exception as e:
            print(f"Error running watch: {e}")
        finally:
            # Reinitialize pygame
            self.init_pygame()

    def run(self):
        """Main loop"""
        try:
            self.main_menu()
        finally:
            pygame.quit()


def main():
    """Entry point"""
    try:
        menu = Menu()
        menu.run()
    except Exception as e:
        print(f"Error in menu: {e}")
        pygame.quit()


if __name__ == "__main__":
    main()

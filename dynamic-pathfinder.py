
import pygame
import heapq
import time
import random
import math
from collections import deque

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
RED = (231, 76, 60)          # Visited nodes
BLUE = (52, 152, 219)        # Start
GREEN = (46, 204, 113)       # Goal / Path
YELLOW = (241, 196, 15)      # Frontier
PURPLE = (155, 89, 182)      # Agent
ORANGE = (230, 126, 34)      # Path
OBSTACLE = (44, 62, 80)      # Obstacles
LIGHT_GRAY = (240, 240, 240)

# Default settings
DEFAULT_ROWS = 20
DEFAULT_COLS = 30
CELL_SIZE = 30
FPS = 60


class GridEnvironment:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = None
        self.goal = None

    def is_valid(self, pos):
        """Check if position is within bounds and not an obstacle"""
        r, c = pos
        return (0 <= r < self.rows and
                0 <= c < self.cols and
                self.grid[r][c] != 1)

    def get_neighbors(self, pos):
        """Get valid 8-directional neighbors"""
        r, c = pos
        directions = [
            (-1, 0),  # Up
            (0, 1),   # Right
            (1, 0),   # Down
            (0, -1),  # Left
            (-1, 1),  # Up-Right
            (-1, -1),  # Up-Left
            (1, 1),   # Down-Right
            (1, -1),  # Down-Left
        ]
        neighbors = []
        for dr, dc in directions:
            new_pos = (r + dr, c + dc)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def set_cell(self, pos, value):
        """Set cell value (0=free, 1=obstacle)"""
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r][c] = value

    def generate_random_obstacles(self, density):
        """Generate random obstacles with given density (0.0 to 1.0)"""
        # Clear existing obstacles
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = 0

        # Generate obstacles
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != self.start and (r, c) != self.goal:
                    if random.random() < density:
                        self.grid[r][c] = 1


def manhattan_distance(pos1, pos2):
    """Manhattan distance: |x1-x2| + |y1-y2|"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1, pos2):
    """Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class SearchAlgorithm:
    def __init__(self, env, heuristic_func):
        self.env = env
        self.heuristic = heuristic_func
        self.nodes_visited = 0
        self.path_cost = 0
        self.execution_time = 0

    def get_path_cost(self, pos1, pos2):
        """Calculate movement cost (1.0 for straight, 1.414 for diagonal)"""
        if pos1[0] == pos2[0] or pos1[1] == pos2[1]:
            return 1.0
        return math.sqrt(2)  # ~1.414

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def a_star(self, start, goal):
        """
        A* Search Algorithm
        f(n) = g(n) + h(n)
        """
        start_time = time.time()
        self.nodes_visited = 0

        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        visited = set()
        frontier = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)
            frontier.discard(current)
            self.nodes_visited += 1

            if current == goal:
                self.execution_time = (time.time() - start_time) * 1000
                path = self.reconstruct_path(came_from, current)
                self.path_cost = g_score[current]
                return path, visited, frontier

            for neighbor in self.env.get_neighbors(current):
                if neighbor in visited:
                    continue

                tentative_g = g_score[current] + \
                    self.get_path_cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + \
                        self.heuristic(neighbor, goal)

                    counter += 1
                    heapq.heappush(
                        open_set, (f_score[neighbor], counter, neighbor))
                    frontier.add(neighbor)

        self.execution_time = (time.time() - start_time) * 1000
        return None, visited, frontier

    def greedy_best_first(self, start, goal):
        """
        Greedy Best-First Search
        f(n) = h(n) only
        """
        start_time = time.time()
        self.nodes_visited = 0

        counter = 0
        open_set = [(self.heuristic(start, goal), counter, start)]
        came_from = {}

        visited = set()
        frontier = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)
            frontier.discard(current)
            self.nodes_visited += 1

            if current == goal:
                self.execution_time = (time.time() - start_time) * 1000
                path = self.reconstruct_path(came_from, current)
                self.path_cost = 0
                for i in range(len(path) - 1):
                    self.path_cost += self.get_path_cost(path[i], path[i+1])
                return path, visited, frontier

            for neighbor in self.env.get_neighbors(current):
                if neighbor in visited:
                    continue

                if neighbor not in came_from:
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (self.heuristic(
                        neighbor, goal), counter, neighbor))
                    frontier.add(neighbor)

        self.execution_time = (time.time() - start_time) * 1000
        return None, visited, frontier


class Button:
    def __init__(self, x, y, width, height, text, color, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.active = False

    def draw(self, screen, font):
        color = tuple(min(c + 30, 255)
                      for c in self.color) if self.active else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class PathfindingApp:
    def __init__(self):
        pygame.init()

        # Settings
        self.rows = DEFAULT_ROWS
        self.cols = DEFAULT_COLS
        self.cell_size = CELL_SIZE

        # Window setup - INCREASED HEIGHT FOR BETTER VISIBILITY
        self.panel_width = 320
        self.grid_width = self.cols * self.cell_size
        self.grid_height = self.rows * self.cell_size
        self.width = self.grid_width + self.panel_width
        self.height = max(self.grid_height, 750)  # Ensure minimum height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(
            "Dynamic Pathfinding Agent - A* & Greedy Best-First")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 18)

        # Environment
        self.env = GridEnvironment(self.rows, self.cols)
        self.env.start = (self.rows // 2, 1)
        self.env.goal = (self.rows // 2, self.cols - 2)

        # State
        self.drawing_mode = "obstacle"
        self.algorithm = "A*"
        self.heuristic = "Manhattan"
        self.dynamic_mode = False
        self.dynamic_probability = 0.02

        self.visualization_state = {
            'visited': set(),
            'frontier': set(),
            'path': []
        }

        self.is_dragging = False
        self.animation_running = False
        self.agent_position = None
        self.current_path = None
        self.path_index = 0
        self.last_move_time = 0
        self.move_delay = 200

        # Metrics
        self.metrics = {
            'nodes_visited': 0,
            'path_cost': 0.0,
            'execution_time': 0.0
        }

        # Setup UI
        self.setup_buttons()

    def setup_buttons(self):
        """Setup all UI buttons with proper spacing"""
        x = self.grid_width + 10
        y = 20
        w = 300
        h = 35
        gap = 8

        self.buttons = []

        # Section: Drawing Modes
        self.buttons.append(Button(x, y, w, h, "Draw Obstacles", OBSTACLE))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "Set Start", BLUE))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "Set Goal", GREEN))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "Erase", WHITE, BLACK))
        y += h + gap * 2

        # Section: Algorithm & Heuristic
        self.buttons.append(Button(x, y, w, h, "Algorithm: A*", PURPLE))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "Heuristic: Manhattan", ORANGE))
        y += h + gap * 2

        # Section: Map Actions
        self.buttons.append(
            Button(x, y, w, h, "Generate Random Maze", DARK_GRAY))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "Clear Obstacles", DARK_GRAY))
        y += h + gap * 2

        # Section: Search Control
        self.buttons.append(Button(x, y, w, h, "▶ Start Search", GREEN))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "⏹ Stop", RED))
        y += h + gap
        self.buttons.append(Button(x, y, w, h, "Reset View", DARK_GRAY))
        y += h + gap * 2

        # Section: Dynamic Mode (WITH PROPER SPACING)
        self.buttons.append(Button(x, y, w, h, "Dynamic Mode: OFF", ORANGE))

        # Update active states
        self.buttons[0].active = True

    def handle_button_click(self, pos):
        """Handle button clicks"""
        for i, button in enumerate(self.buttons):
            if button.is_clicked(pos):
                if i == 0:
                    self.drawing_mode = "obstacle"
                    self.set_active_button(0, 0, 4)
                elif i == 1:
                    self.drawing_mode = "start"
                    self.set_active_button(1, 0, 4)
                elif i == 2:
                    self.drawing_mode = "goal"
                    self.set_active_button(2, 0, 4)
                elif i == 3:
                    self.drawing_mode = "erase"
                    self.set_active_button(3, 0, 4)
                elif i == 4:
                    self.algorithm = "GBFS" if self.algorithm == "A*" else "A*"
                    self.buttons[4].text = f"Algorithm: {self.algorithm}"
                elif i == 5:
                    self.heuristic = "Euclidean" if self.heuristic == "Manhattan" else "Manhattan"
                    self.buttons[5].text = f"Heuristic: {self.heuristic}"
                elif i == 6:
                    self.generate_random_maze()
                elif i == 7:
                    self.clear_obstacles()
                elif i == 8:
                    self.start_search()
                elif i == 9:
                    self.stop_animation()
                elif i == 10:
                    self.reset_view()
                elif i == 11:
                    self.dynamic_mode = not self.dynamic_mode
                    self.buttons[11].text = f"Dynamic Mode: {'ON' if self.dynamic_mode else 'OFF'}"
                    self.buttons[11].active = self.dynamic_mode

    def set_active_button(self, active_idx, start_idx, end_idx):
        """Set one button active in a range"""
        for i in range(start_idx, end_idx):
            self.buttons[i].active = (i == active_idx)

    def handle_grid_click(self, pos):
        """Handle clicks on the grid"""
        x, y = pos
        if x >= self.grid_width:
            return

        col = x // self.cell_size
        row = y // self.cell_size

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return

        grid_pos = (row, col)

        if self.drawing_mode == "start":
            self.env.start = grid_pos
        elif self.drawing_mode == "goal":
            self.env.goal = grid_pos
        elif self.drawing_mode == "obstacle":
            if grid_pos != self.env.start and grid_pos != self.env.goal:
                self.env.set_cell(grid_pos, 1)
        elif self.drawing_mode == "erase":
            if grid_pos != self.env.start and grid_pos != self.env.goal:
                self.env.set_cell(grid_pos, 0)

    def generate_random_maze(self):
        """Generate random maze"""
        self.env.generate_random_obstacles(0.3)
        self.reset_view()

    def clear_obstacles(self):
        """Clear all obstacles"""
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != self.env.start and (r, c) != self.env.goal:
                    self.env.set_cell((r, c), 0)
        self.reset_view()

    def reset_view(self):
        """Reset visualization"""
        self.visualization_state = {
            'visited': set(),
            'frontier': set(),
            'path': []
        }
        self.metrics = {
            'nodes_visited': 0,
            'path_cost': 0.0,
            'execution_time': 0.0
        }
        self.animation_running = False
        self.agent_position = None

    def start_search(self):
        """Start pathfinding search"""
        if not self.env.start or not self.env.goal:
            print("Please set both Start and Goal positions!")
            return

        if self.animation_running:
            return

        self.reset_view()

        heuristic_func = manhattan_distance if self.heuristic == "Manhattan" else euclidean_distance
        searcher = SearchAlgorithm(self.env, heuristic_func)

        if self.algorithm == "A*":
            result = searcher.a_star(self.env.start, self.env.goal)
        else:
            result = searcher.greedy_best_first(self.env.start, self.env.goal)

        path, visited, frontier = result

        if path is None:
            print("No path found!")
            return

        self.visualization_state['visited'] = visited
        self.visualization_state['frontier'] = frontier
        self.visualization_state['path'] = path

        self.metrics['nodes_visited'] = searcher.nodes_visited
        self.metrics['path_cost'] = searcher.path_cost
        self.metrics['execution_time'] = searcher.execution_time

        if self.dynamic_mode:
            self.animation_running = True
            self.current_path = path
            self.path_index = 0
            self.agent_position = path[0]
            self.last_move_time = pygame.time.get_ticks()

    def stop_animation(self):
        """Stop animation"""
        self.animation_running = False
        self.agent_position = None

    def update_animation(self):
        """Update agent animation"""
        if not self.animation_running or not self.current_path:
            return

        current_time = pygame.time.get_ticks()
        if current_time - self.last_move_time < self.move_delay:
            return

        self.last_move_time = current_time

        if self.path_index >= len(self.current_path):
            self.animation_running = False
            print("Goal reached!")
            return

        self.agent_position = self.current_path[self.path_index]

        if random.random() < self.dynamic_probability:
            self.spawn_dynamic_obstacle()

        if self.path_index + 1 < len(self.current_path):
            next_pos = self.current_path[self.path_index + 1]
            if self.env.grid[next_pos[0]][next_pos[1]] == 1:
                self.replan_from_current()
                return

        self.path_index += 1

    def spawn_dynamic_obstacle(self):
        """Spawn a random obstacle"""
        for _ in range(5):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            pos = (r, c)

            if (pos != self.env.start and
                pos != self.env.goal and
                self.env.grid[r][c] == 0 and
                    pos not in self.current_path[max(0, self.path_index-1):self.path_index+3]):

                self.env.set_cell(pos, 1)
                print(f"Dynamic obstacle spawned at {pos}")
                break

    def replan_from_current(self):
        """Replan path from current position"""
        print("Path blocked! Replanning...")
        heuristic_func = manhattan_distance if self.heuristic == "Manhattan" else euclidean_distance
        searcher = SearchAlgorithm(self.env, heuristic_func)

        if self.algorithm == "A*":
            result = searcher.a_star(self.agent_position, self.env.goal)
        else:
            result = searcher.greedy_best_first(
                self.agent_position, self.env.goal)

        path, visited, frontier = result

        if path is None:
            print("Cannot find alternative path!")
            self.animation_running = False
            return

        self.visualization_state['visited'].update(visited)
        self.visualization_state['frontier'] = frontier
        self.visualization_state['path'] = path

        self.metrics['nodes_visited'] += searcher.nodes_visited
        self.metrics['path_cost'] = searcher.path_cost

        self.current_path = path
        self.path_index = 0
        print("New path found!")

    def draw_grid(self):
        """Draw the grid"""
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size
                y = r * self.cell_size
                pos = (r, c)

                if pos == self.env.start:
                    color = BLUE
                elif pos == self.env.goal:
                    color = GREEN
                elif pos == self.agent_position:
                    color = PURPLE
                elif pos in self.visualization_state['path']:
                    color = ORANGE
                elif pos in self.visualization_state['visited']:
                    color = RED
                elif pos in self.visualization_state['frontier']:
                    color = YELLOW
                elif self.env.grid[r][c] == 1:
                    color = OBSTACLE
                else:
                    color = WHITE

                pygame.draw.rect(self.screen, color,
                                 (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, GRAY,
                                 (x, y, self.cell_size, self.cell_size), 1)

    def draw_panel(self):
        """Draw control panel"""
        x = self.grid_width
        pygame.draw.rect(self.screen, LIGHT_GRAY,
                         (x, 0, self.panel_width, self.height))

        # Draw all buttons
        for button in self.buttons:
            button.draw(self.screen, self.font_small)

        # Draw metrics at bottom
        y = self.height - 140
        x_offset = self.grid_width + 10

        # Metrics background
        pygame.draw.rect(self.screen, WHITE, (x_offset,
                         y, 300, 120), border_radius=5)
        pygame.draw.rect(self.screen, BLACK, (x_offset,
                         y, 300, 120), 2, border_radius=5)
        y += 10

        title = self.font_medium.render("Real-Time Metrics", True, BLACK)
        self.screen.blit(title, (x_offset + 10, y))
        y += 30

        visited_text = self.font_small.render(
            f"Nodes Visited: {self.metrics['nodes_visited']}", True, BLACK)
        self.screen.blit(visited_text, (x_offset + 10, y))
        y += 25

        cost_text = self.font_small.render(
            f"Path Cost: {self.metrics['path_cost']:.2f}", True, BLACK)
        self.screen.blit(cost_text, (x_offset + 10, y))
        y += 25

        time_text = self.font_small.render(
            f"Time: {self.metrics['execution_time']:.2f} ms", True, BLACK)
        self.screen.blit(time_text, (x_offset + 10, y))

    def run(self):
        """Main game loop"""
        running = True

        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.is_dragging = True
                        self.handle_button_click(event.pos)
                        self.handle_grid_click(event.pos)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.is_dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.is_dragging:
                        self.handle_grid_click(event.pos)

            self.update_animation()

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_panel()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    app = PathfindingApp()
    app.run()

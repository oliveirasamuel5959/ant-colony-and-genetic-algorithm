import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from enum import Enum
from typing import List, Optional, Dict, Any
from solvers.genetic_algorithm import GeneticAlgorithm
from solvers.aco_solver import AntColonySolver
from data.loader import generate_random_cities, load_from_json

class State(Enum):
    SETUP = 1
    GA = 2
    ACO = 3
    COMPARISON = 4

class UIManager:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.state = State.SETUP
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 32, bold=True)
        
        # Dados e Solvers - Carrega do arquivo JSON conforme solicitado
        self.cities: np.ndarray = load_from_json("data/cities.json")
        
        # Fallback caso o arquivo falhe
        if self.cities.size == 0:
            self.cities = generate_random_cities(10, self.width - 200, self.height - 300)
            
        self.ga_solver: Optional[GeneticAlgorithm] = None
        self.aco_solver: Optional[AntColonySolver] = None
        
        # Hiperparâmetros padrão
        self.params = {
            "pop_size": 100,
            "mutation_rate": 0.05,
            "num_ants": 10,
            "alpha": 1.0,
            "beta": 2.0,
            "evaporation": 0.1
        }
        
        self.running_simulation = False

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1: self.state = State.SETUP
            if event.key == pygame.K_2: self.setup_solvers(State.GA)
            if event.key == pygame.K_3: self.setup_solvers(State.ACO)
            if event.key == pygame.K_4: self.setup_solvers(State.COMPARISON)
            if event.key == pygame.K_SPACE: self.running_simulation = not self.running_simulation
            if event.key == pygame.K_r: self.reset_simulation()
            if event.key == pygame.K_g: self.generate_new_dataset(20)

    def generate_new_dataset(self, num_cities: int):
        """Gera um novo dataset, salva no JSON e recarrega na memória."""
        generate_random_cities(num_cities, self.width - 200, self.height - 300, save_path="data/cities.json")
        self.cities = load_from_json("data/cities.json")
        self.reset_simulation()
        print(f"Novo dataset de {num_cities} cidades gerado.")

    def setup_solvers(self, state: State):
        self.state = state
        self.running_simulation = False
        if state in [State.GA, State.COMPARISON]:
            self.ga_solver = GeneticAlgorithm(self.cities, self.params)
        if state in [State.ACO, State.COMPARISON]:
            self.aco_solver = AntColonySolver(self.cities, self.params)

    def reset_simulation(self):
        self.setup_solvers(self.state)

    def on_resize(self, width: int, height: int):
        self.width = width
        self.height = height

    def update(self):
        if self.running_simulation:
            if self.state == State.GA and self.ga_solver:
                self.ga_solver.evolve()
            elif self.state == State.ACO and self.aco_solver:
                self.aco_solver.evolve()
            elif self.state == State.COMPARISON:
                if self.ga_solver: self.ga_solver.evolve()
                if self.aco_solver: self.aco_solver.evolve()

    def draw(self):
        self.screen.fill((20, 20, 25))  # Cor de fundo ligeiramente mais escura e moderna
        
        if self.state == State.SETUP:
            self._draw_setup()
        elif self.state == State.GA:
            self._draw_solver_state(self.ga_solver, "Algoritmo Genético", (0, 0, self.width, self.height))
        elif self.state == State.ACO:
            self._draw_solver_state(self.aco_solver, "Colônia de Formigas", (0, 0, self.width, self.height), draw_pheromones=True)
        elif self.state == State.COMPARISON:
            # Divisória central
            pygame.draw.line(self.screen, (60, 60, 70), (self.width // 2, 50), (self.width // 2, self.height - 100), 2)
            self._draw_solver_state(self.ga_solver, "Algoritmo Genético (GA)", (0, 0, self.width//2, self.height))
            self._draw_solver_state(self.aco_solver, "Colônia de Formigas (ACO)", (self.width//2, 0, self.width//2, self.height), draw_pheromones=True)
        
        self._draw_footer()
        pygame.display.flip()

    def _draw_setup(self):
        title = self.title_font.render("Configuração do Simulador TSP", True, (0, 200, 255))
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 80))
        
        instructions = [
            "Controles de Navegação:",
            "  [1] Menu Inicial (Setup)",
            "  [2] Visualizar Algoritmo Genético",
            "  [3] Visualizar Colônia de Formigas",
            "  [4] Modo Comparação Lado a Lado",
            "",
            "Controles de Execução:",
            "  [ESPAÇO] Iniciar / Pausar Evolução",
            "  [R] Reiniciar Solvers Atuais",
            "  [G] Gerar Novo Dataset (20 cidades)",
            "",
            "Configurações Atuais:",
            f"  Cidades no Dataset: {len(self.cities)}",
            f"  Parâmetros GA: Pop: {self.params['pop_size']}, Mut: {self.params['mutation_rate']*100}%",
            f"  Parâmetros ACO: Formigas: {self.params['num_ants']}, α: {self.params['alpha']}, β: {self.params['beta']}"
        ]
        
        for i, text in enumerate(instructions):
            color = (255, 255, 255) if text.endswith(":") else (180, 180, 180)
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (self.width//2 - 250, 200 + i*35))

    def _draw_solver_state(self, solver: Any, title: str, rect: tuple, draw_pheromones=False):
        x, y, w, h = rect
        padding = 40
        
        # Título do solver
        t_surf = self.title_font.render(title, True, (255, 255, 255))
        self.screen.blit(t_surf, (x + w//2 - t_surf.get_width()//2, y + 40))
        
        # Ajuste dinâmico da área de desenho
        graph_h = 120
        stats_h = 100
        draw_area = (x + padding, y + 120, w - 2*padding, h - graph_h - stats_h - 180)
        
        if solver:
            if draw_pheromones and hasattr(solver, 'get_pheromones'):
                self._draw_pheromones(solver, draw_area)
            
            self._draw_path(solver, draw_area)
            self._draw_cities(solver.cities, draw_area)
            self._draw_stats(solver, (x + padding, h - graph_h - stats_h - 30))
            self._draw_graph(solver.get_history(), (x + w//2, h - 80), w - 2*padding, graph_h)

    def _draw_cities(self, cities: np.ndarray, area: tuple):
        ax, ay, aw, ah = area
        # Encontra limites para normalização dentro da área
        min_c = np.min(cities, axis=0)
        max_c = np.max(cities, axis=0)
        range_c = max_c - min_c
        
        for i, city in enumerate(cities):
            nx = (city[0] - min_c[0]) / range_c[0] if range_c[0] > 0 else 0.5
            ny = (city[1] - min_c[1]) / range_c[1] if range_c[1] > 0 else 0.5
            
            px = ax + nx * aw
            py = ay + ny * ah
            
            # Sombra da cidade
            pygame.draw.circle(self.screen, (0, 0, 0), (int(px)+2, int(py)+2), 7)
            pygame.draw.circle(self.screen, (255, 60, 60), (int(px), int(py)), 6)
            # Brilho
            pygame.draw.circle(self.screen, (255, 200, 200), (int(px)-1, int(py)-1), 2)

            # Rótulo da cidade (1, 2, 3...)
            label = self.font.render(str(i + 1), True, (255, 255, 255))
            self.screen.blit(label, (int(px) + 10, int(py) - 15))

    def _draw_path(self, solver: Any, area: tuple):
        path = solver.get_best_path()
        if not path: return
        
        ax, ay, aw, ah = area
        cities = solver.cities
        min_c = np.min(cities, axis=0)
        max_c = np.max(cities, axis=0)
        range_c = max_c - min_c
        
        points = []
        for idx in path:
            city = cities[idx]
            nx = (city[0] - min_c[0]) / range_c[0] if range_c[0] > 0 else 0.5
            ny = (city[1] - min_c[1]) / range_c[1] if range_c[1] > 0 else 0.5
            points.append((ax + nx * aw, ay + ny * ah))
        
        if len(points) > 1:
            # Desenha caminho com um efeito de glow suave
            pygame.draw.lines(self.screen, (0, 100, 0), True, points, 5)
            pygame.draw.lines(self.screen, (0, 255, 100), True, points, 2)

    def _draw_pheromones(self, solver: AntColonySolver, area: tuple):
        ph = solver.get_pheromones()
        ax, ay, aw, ah = area
        cities = solver.cities
        min_c = np.min(cities, axis=0)
        max_c = np.max(cities, axis=0)
        range_c = max_c - min_c
        
        max_ph = np.max(ph) if np.max(ph) > 0 else 1.0
        
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                strength = ph[i, j] / max_ph
                if strength > 0.05:
                    c1, c2 = cities[i], cities[j]
                    n1 = (c1 - min_c) / range_c
                    n2 = (c2 - min_c) / range_c
                    p1 = (ax + n1[0]*aw, ay + n1[1]*ah)
                    p2 = (ax + n2[0]*aw, ay + n2[1]*ah)
                    
                    # Interpolação de cor para feromônio
                    color_val = int(strength * 150)
                    pygame.draw.line(self.screen, (50, 50, 100 + color_val), p1, p2, max(1, int(strength * 4)))

    def _draw_stats(self, solver: Any, pos: tuple):
        path = solver.get_best_path()
        dist = solver.calculate_total_distance(path) if path else 0
        iter_count = len(solver.get_history())
        
        stats = [
            f"Melhor Distância: {dist:.2f}",
            f"Iteração/Geração: {iter_count}"
        ]
        
        for i, text in enumerate(stats):
            surf = self.font.render(text, True, (0, 255, 150))
            self.screen.blit(surf, (pos[0], pos[1] + i*25))

    def _draw_graph(self, history: List[float], pos: tuple, w: int, h: int):
        if len(history) < 2: return
        
        cx, cy = pos
        pts = []
        max_val = max(history)
        min_val = min(history)
        rng = max_val - min_val if max_val != min_val else 1.0
        
        # Fundo do gráfico
        graph_rect = (cx - w//2, cy - h//2, w, h)
        pygame.draw.rect(self.screen, (40, 40, 50), graph_rect)
        pygame.draw.rect(self.screen, (70, 70, 80), graph_rect, 1)
        
        for i, val in enumerate(history):
            x = cx - w//2 + (i / (len(history) - 1)) * w
            y = cy + h//2 - ((val - min_val) / rng) * h
            pts.append((x, y))
            
        pygame.draw.lines(self.screen, (255, 200, 0), False, pts, 2)
        
        # Rótulos de min/max
        min_text = self.font.render(f"{min_val:.0f}", True, (150, 150, 150))
        self.screen.blit(min_text, (cx + w//2 + 5, cy + h//2 - 10))

    def _draw_footer(self):
        footer = self.font.render("FEI - TSP Simulator | 1:Setup 2:GA 3:ACO 4:Comp | ESPAÇO:Start/Pause R:Reset", True, (150, 150, 150))
        self.screen.blit(footer, (20, self.height - 30))

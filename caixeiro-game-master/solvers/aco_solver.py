"""
Centro Universitário FEI - Departamento de Engenharia Elétrica
Disciplina: PEL 202 - Introdução a Inteligência Artificial
Projeto: Resolução do TSP via Colônia de Formigas (ACO)
Professor: Rafael Gomes Alves
Aluno(a): ______________________________________
Data: 2026
Descrição: Implementação de lógica de otimização para o Caixeiro Viajante.
"""

from core.base_solver import TSPSolver
import numpy as np
from typing import List, Tuple

class AntColonySolver(TSPSolver):
    def __init__(self, cities: np.ndarray, params: dict):
        """Inicializa o solver com os parâmetros da Colônia de Formigas.
        Args:
            - cities (np.ndarray): Matriz de coordenadas das cidades.
            - params (dict): Dicionário de parâmetros do algoritmo (ex: num_ants, alpha, beta, evaporation).
        Returns:
            - None
        """
        super().__init__(cities, params)
        self.num_ants = params.get("num_ants", 20)
        self.alpha = params.get("alpha", 1.0)      # Importância do feromônio
        self.beta = params.get("beta", 2.0)       # Importância da visibilidade (1/dist)
        self.rho = params.get("evaporation", 0.1) # Taxa de evaporação
        self.q = params.get("q_factor", 100.0)    # Constante de depósito
        
        # Matriz de distâncias
        self.dist_matrix = self._compute_dist_matrix()
        # Matriz de feromônios inicializada com valor pequeno
        self.pheromone = np.ones((self.num_cities, self.num_cities)) * 0.1

    def _compute_dist_matrix(self) -> np.ndarray:
        """Calcula a matriz de distâncias entre as cidades.
        Args:
            - None
        Returns:
            - np.ndarray: Matriz de distâncias.
        """
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    matrix[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return matrix

    def evolve(self) -> None:
        """
        Executa uma iteração da Colônia de Formigas.
        Args:
            - None
        Returns:
            - None
        """
        # 1. Construção das soluções (movimentação das formigas)
        # 2. Atualização local/global dos feromônios
        # 3. Evaporação
        
        # TODO: Implementar lógica do aluno
        # Dica: Atualize self.best_path, self.best_distance e self.history
        pass

    def _select_next_city(self, current_city: int, visited: set) -> int:
        """
        Escolhe a próxima cidade baseada na probabilidade (equação do ACO).
        Args:
            - current_city (int): Cidade atual da formiga.
            - visited (set): Conjunto de cidades já visitadas pela formiga.
        Returns:
            - int: Próxima cidade selecionada.
        """
        # TODO: Implementar lógica do aluno
        return 0

    def _update_pheromones(self, all_paths: List[List[int]], distances: List[float]) -> None:
        """
        Atualiza a matriz de feromônios baseada nos caminhos encontrados.
        Args:
            - all_paths (List[List[int]]): Lista de caminhos percorridos pelas formigas.
            - distances (List[float]): Lista de distâncias correspondentes a cada caminho.
        Returns:
            - None
        """
        # TODO: Implementar lógica do aluno
        pass
        
    def get_pheromones(self) -> np.ndarray:
        """Retorna a matriz de feromônios para visualização.
        Args:
            - None
        Returns:
            - np.ndarray: Matriz de feromônios.
        """
        return self.pheromone

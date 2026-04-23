from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np

class TSPSolver(ABC):
    """
    Classe base abstrata para solucionadores do Problema do Caixeiro Viajante (TSP).
    """
    def __init__(self, cities: np.ndarray, params: dict):
        """
        Inicializa o solver com as coordenadas das cidades e hiperparâmetros.
        
        Args:
            cities: Array numpy de shape (N, 2) com as coordenadas (x, y).
            params: Dicionário contendo os hiperparâmetros do algoritmo.
        """
        self.cities = cities
        self.params = params
        self.num_cities = len(cities)
        self.best_path: List[int] = []
        self.best_distance: float = float('inf')
        self.history: List[float] = []

    def calculate_total_distance(self, path: List[int]) -> float:
        """Calcula a distância total de um caminho fechado."""
        if not path:
            return float('inf')
        
        distance = 0.0
        for i in range(len(path)):
            c1 = self.cities[path[i]]
            c2 = self.cities[path[(i + 1) % len(path)]]
            distance += np.linalg.norm(c1 - c2)
        return distance

    @abstractmethod
    def evolve(self) -> None:
        """Executa uma iteração (geração/iteração de formigas) do algoritmo."""
        pass

    def get_best_path(self) -> List[int]:
        """Retorna o melhor caminho encontrado até o momento."""
        return self.best_path

    def get_history(self) -> List[float] :
        """Retorna o histórico de evolução do custo (distância)."""
        return self.history

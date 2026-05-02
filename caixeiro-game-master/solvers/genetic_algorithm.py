"""
Centro Universitário FEI - Departamento de Engenharia Elétrica
Disciplina: PEL 202 - Introdução a Inteligência Artificial
Projeto: Resolução do TSP via Algoritmo Genético
Professor: Rafael Gomes Alves
Aluno(a): ______________________________________
Data: 2026
Descrição: Implementação de lógica de otimização para o Caixeiro Viajante.
"""

from core.base_solver import TSPSolver
import numpy as np
from typing import List, Tuple

class GeneticAlgorithm(TSPSolver):
    def __init__(self, cities: np.ndarray, params: dict):
        """Inicializa o solver com os parâmetros do Algoritmo Genético.
        Args:
            - cities (np.ndarray): Matriz de coordenadas das cidades.
            - params (dict): Dicionário de parâmetros do algoritmo (ex: pop_size, mutation_rate).
        Returns:
            - None
        """
        super().__init__(cities, params)
        self.pop_size = params.get("pop_size", 100)
        self.mutation_rate = params.get("mutation_rate", 0.01)
        self.population = self._init_population()
        
    def _init_population(self) -> List[List[int]]:
        """Gera uma população inicial aleatória.
        A população é composta por indivíduos, onde cada indivíduo é uma permutação das cidades representando um caminho.
        Exemplo: Se temos 5 cidades, um indivíduo pode ser [0, 2, 4, 1, 3], indicando a ordem de visitação das cidades.
        Args:
            - None
        Returns: 
            - List[List[int]]: Lista de indivíduos (caminhos) na população.
        """
        pop = []
        for _ in range(self.pop_size): 
            individual = list(range(self.num_cities)) #
            np.random.shuffle(individual) 
            pop.append(individual)
        return pop 

    def evolve(self) -> None:
        """
        Executa uma geração do Algoritmo Genético.
        """
        # 1. Calculate fitness scores for current population
        fitness_scores = [self._calc_fitness(ind) for ind in self.population]
        
        # 2. Create new population through selection, crossover, and mutation
        new_population = []
        while len(new_population) < self.pop_size:
            parent1 = self.tournament_selection(fitness_scores)
            parent2 = self.tournament_selection(fitness_scores)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        # 3. Replace old population
        self.population = new_population
        
        # 4. Update best path and track history
        for ind in self.population:
            distance = self.calculate_total_distance(ind)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = ind
        self.history.append(self.best_distance)
        
    def _calc_fitness(self, individual: List[int]) -> float:
        """Rank-based or linear scaling"""
        distance = self.calculate_total_distance(individual)
        # Linear scaling: map to positive range
        max_dist = max(self.calculate_total_distance(ind) for ind in self.population)
        return (max_dist - distance + 1) / (max_dist + 1)
    
    def tournament_selection(self, score: List[float], k: int = 3) -> List[int]:
        """
        Realiza a seleção por torneio.
        Args:
            - score (List[float]): Lista de fitness dos indivíduos.
            - k (int): Número de indivíduos a serem selecionados para o torneio.
        Returns:
            - List[int]: Índice do indivíduo selecionado.
        """
        selected_indices = np.random.choice(len(self.population), k, replace=False)
        selected_scores = [score[i] for i in selected_indices]
        winner_index = selected_indices[np.argmax(selected_scores)]
        return self.population[winner_index]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover - TSP-specific, preserves tour order"""
        size = len(parent1)
        idx1, idx2 = sorted(np.random.choice(size, 2, replace=False))
        
        child = [-1] * size
        # Copy segment from parent1
        child[idx1:idx2] = parent1[idx1:idx2]
        
        # Fill remaining from parent2 in order
        p2_idx = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
        
        return child
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """
        Aplica mutação em um indivíduo (ex: Swap Mutation).
        Args:
            - individual (List[int]): Indivíduo a ser mutado.
        Returns:
            - List[int]: Indivíduo mutado.
        """
        # TODO: Implementar lógica do aluno

        # Dica: A mutação deve ocorrer com uma probabilidade definida por self.mutation_rate.
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual

# if __name__ == "__main__":
#     # Exemplo de uso
#     cities = np.array([[0, 1, 2, 3, 4], [1, 4, 2, 3, 4], [2, 3, 1, 0, 4], [4, 0, 1, 2, 3], [3, 0, 1, 4, 2]])
#     params = {"pop_size": 50, "mutation_rate": 0.05}
#     solver = GeneticAlgorithm(cities, params)
#     solver.evolve()
#     print(f"Melhor distância = {solver.best_distance}")
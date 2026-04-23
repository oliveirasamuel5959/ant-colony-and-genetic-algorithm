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
        Args:
            - None
        Returns:
            - None
        """
        # 1. Avaliação de Fitness
        # 2. Seleção
        # 3. Cruzamento (Crossover)
        # 4. Mutação
        
        # TODO: Implementar lógica do aluno
        # Dica: Atualize self.best_path, self.best_distance e self.history
        pass

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Realiza o cruzamento entre dois pais para gerar um filho.
        Utilize operadores adequados para o TSP (ex: Order Crossover).
        Args:
            - parent1 (List[int]): Primeiro pai.
            - parent2 (List[int]): Segundo pai.
        Returns:
            - List[int]: Filho gerado.
        """
        # TODO: Implementar lógica do aluno
        return parent1

    def _mutate(self, individual: List[int]) -> List[int]:
        """
        Aplica mutação em um indivíduo (ex: Swap Mutation).
        Args:
            - individual (List[int]): Indivíduo a ser mutado.
        Returns:
            - List[int]: Indivíduo mutado.
        """
        # TODO: Implementar lógica do aluno
        return individual

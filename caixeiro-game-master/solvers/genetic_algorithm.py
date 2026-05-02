"""
Centro Universitário FEI - Departamento de Engenharia Elétrica
Disciplina: PEL 202 - Introdução a Inteligência Artificial
Projeto: Resolução do TSP via Algoritmo Genético
Professor: Rafael Gomes Alves
Aluno(a): ______________________________________
Data: 2026
Descrição: Implementação de lógica de otimização para o Caixeiro Viajante.
"""

import random

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
        self.mutation_rate = params.get("mutation_rate", 0.05)
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
    
    def _test_fitness(self) -> None:
        """Função de teste para verificar a evolução do algoritmo.
        Esta função executa uma geração do algoritmo e imprime o melhor caminho e distância encontrados.
        Args:
            - None
        Returns:
            - None
        """
        fitness_scores = [self._calc_fitness(ind) for ind in self.population]
        
        # sort population by their fitness score
        order = np.array(sorted([*enumerate(fitness_scores)], key=lambda x: x[1], reverse=True), dtype=int)[:, 0] 
        self.population = [self.population[i] for i in order]
        fitness_scores = sorted(fitness_scores, reverse=True)
        
        for i in range(len(self.population)):
            print(f"Individual {i}: {self.population[i]}, Fitness: {round(fitness_scores[i], 3)}")
            
    def _test_selection(self) -> None:
        """Função de teste para verificar a seleção por torneio.
        Esta função executa a seleção por torneio e imprime o indivíduo selecionado.
        Args:
            - None
        Returns:
            - None
        """
        fitness_scores = [self._calc_fitness(ind) for ind in self.population]
        
        parent1 = self.tournament_selection(fitness_scores)
        parent2 = self.tournament_selection(fitness_scores)
        
        print(f"Selected parent 1:\t {parent1}, Fitness: {round(self._calc_fitness(parent1), 3)}")
        print(f"Selected parent 2:\t {parent2}, Fitness: {round(self._calc_fitness(parent2), 3)}")
    
    def _test_crossover(self) -> None:
        """Função de teste para verificar o crossover.
        Esta função seleciona dois indivíduos, realiza o crossover e imprime o filho gerado.
        Args:
            - None
        Returns:
            - None
        """        
        parent1 = self.population[random.randint(0, self.pop_size - 1)]
        parent2 = self.population[random.randint(0, self.pop_size - 1)]
        
        child = self._crossover(parent1.copy(), parent2.copy())
        
        print(f"Parent 1:\t {parent1}")
        print(f"Parent 2:\t {parent2}")
        print(f"Child:\t\t {child}")
        
    def _test_mutation(self) -> None:
        """Função de teste para verificar a mutação.
        Esta função seleciona um indivíduo, realiza a mutação e imprime o indivíduo mutado.
        Args:
            - None
        Returns:
            - None
        """
        individual = self.population[random.randint(0, self.pop_size - 1)]
        mutated_individual = self._mutate(individual.copy())
        
        print(f"Original Individual:\t {individual}")
        print(f"Mutated Individual:\t {mutated_individual}")
    
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
        '''
        Calcula o fitness de um indivíduo com base na distância total do caminho.
        Args:
            - individual (List[int]): Indivíduo representando um caminho (permutação das cidades).
        Returns:
            - float: Valor de fitness, onde um valor mais alto indica um caminho melhor (menor distância).
        '''
        
        # A função de fitness é baseada na distância total do caminho representado pelo indivíduo.
        distance = self.calculate_total_distance(individual)
        
        # Função de fitness inversa: Quanto menor a distância, maior o fitness
        # max_dist = max(self.calculate_total_distance(ind) for ind in self.population)
        
        return 1 / (distance + 1e-6) if distance > 0 else float('inf') # Evita divisão por zero
    
    def tournament_selection(self, score: List[float], k: int = 3) -> List[int]:
        """
        Realiza a seleção por torneio.
        Args:
            - score (List[float]): Lista de fitness dos indivíduos da população.
            - k (int): Número de indivíduos a serem selecionados para o torneio.
        Returns:
            - List[int]: Índice do indivíduo selecionado.
        """
        
        print(f"\n--- Tournament Selection (k={k}) ---")
        
        # Seleciona k indivíduos da população aleatoriamente para o torneio
        selected_indices = np.random.choice(len(self.population), k, replace=False)
        
        # Obtém os scores dos indivíduos selecionados e determina o vencedor (o de maior fitness)
        selected_scores = [score[i] for i in selected_indices]
        
        # O vencedor é o indivíduo com o maior score (fitness)
        winner_index = selected_indices[np.argmax(selected_scores)]
        
        # 
        # ========================================
        # PRINT RESULTS FOR TESTES USING PYTEST
        # ========================================
        print(f"Selected indices: {selected_indices}")
        print(f"Scores: {[round(s, 3) for s in selected_scores]}")
        print(f"Winner idx: {winner_index}")
        print(f"Winner individual: {self.population[winner_index]}")
        
        # Retorna o indivíduo vencedor do torneio
        return self.population[winner_index]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        '''Realiza o crossover entre dois indivíduos (ex: Order Crossover).
        Args:
            - parent1 (List[int]): Primeiro indivíduo (caminho).
            - parent2 (List[int]): Segundo indivíduo (caminho).
        Returns:
            - List[int]: Filho gerado a partir do crossover dos pais.
        '''
        
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
        
        # 
        # ========================================
        # PRINT RESULTS FOR TESTES USING PYTEST
        # ========================================
        print(f"\n--- Crossover ---")
        print(f"Parent 1 segment [{idx1}:{idx2}]: {parent1[idx1:idx2]}")
        print(f"Parent 2: {parent2}")
        print(f"Child: {child}")
        
        return child
    
    def _mutate(self, individual: List[int]) -> List[int]:
        '''
        Aplica mutação em um indivíduo (ex: Swap Mutation).
        
        Args:
            - individual (List[int]): Indivíduo a ser mutado.
        Returns:
            - List[int]: Indivíduo mutado.
        '''
        print(f"\n--- Mutation ---")
        
        # A mutação deve ocorrer com uma probabilidade definida por self.mutation_rate.
        original = individual.copy()
        
        if random.random() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            # 
            # ========================================
            # PRINT RESULTS FOR TESTES USING PYTEST
            # ========================================
            print(f"✓ MUTATION: {original} → {individual}\n")
        else:
            print(f"✗ NO MUTATION: {individual}\n")
        
        return individual
        
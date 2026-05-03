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
        Implementa estratégia de elitismo (mantém os 5 melhores indivíduos) 
        e seleção por torneio com probabilidades de fitness para gerar novos indivíduos.
        """
        # 1. Calculate distances for all individuals (needed for elitism)
        distances = [self.calculate_total_distance(ind) for ind in self.population]
        
        # 2. Calculate fitness probabilities for tournament selection
        fitness_probs = self._calculate_population_fitness_probabilities()
        
        # 3. Apply elitism - preserve the top 5 elite individuals
        elite_size = 5
        elite_individuals = self._elitism_selection(fitness_probs, elite_size=elite_size)
        
        # 4. Create new population starting with elites
        new_population = [ind.copy() for ind in elite_individuals]
        
        # 5. Fill the rest of population through tournament selection, crossover, and mutation
        while len(new_population) < self.pop_size:
            parent1 = self.tournament_selection(fitness_probs, k=7)
            parent2 = self.tournament_selection(fitness_probs, k=7)
            child = self._crossover(parent1.copy(), parent2.copy())
            child = self._mutate(child)
            new_population.append(child)
        
        # 6. Replace old population
        self.population = new_population
        
        # 7. Update best path and track history
        for ind in self.population:
            distance = self.calculate_total_distance(ind)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = ind
        self.history.append(self.best_distance)
        
        print(f"Best path: {self.best_path}, Distance: {round(self.best_distance, 3)}")
        
    def _calc_fitness(self, individual: List[int]) -> float:
        """Calculate fitness for a single individual based on distance."""
        distance = self.calculate_total_distance(individual)
        return distance
    
    def _calculate_population_fitness_probabilities(self) -> np.ndarray:
        """
        Calculates fitness probabilities for the entire population.
        Uses inverse distance scaling: fitness = max_distance - individual_distance
        Normalizes to create probability distribution across population.
        
        Returns:
            np.ndarray: Array of fitness probabilities for each individual in population
        """
        # Calculate total distance for all individuals
        total_distances = np.array([self.calculate_total_distance(ind) for ind in self.population])
        
        # Find max distance (worst individual)
        max_distance = np.max(total_distances)
        
        # Calculate fitness: max_distance - individual_distance (higher is better)
        population_fitness = max_distance - total_distances
        
        # Normalize to create probabilities (sum = 1.0)
        fitness_sum = np.sum(population_fitness)
        if fitness_sum > 0:
            population_fitness_probs = population_fitness / fitness_sum
        else:
            # Fallback if all fitness values are equal
            population_fitness_probs = np.ones(len(self.population)) / len(self.population)
        
        return population_fitness_probs
        
    def _elitism_selection(self, score: List[float], elite_size: int = 10) -> List[List[int]]:
        """
        Realiza a seleção por elitismo, onde os melhores indivíduos são automaticamente selecionados para a próxima geração.
        Args:
            - score (List[float]): Lista de fitness dos indivíduos da população.
            - elite_size (int): Número de indivíduos a serem mantidos como elite.
        Returns:
            - List[List[int]]: Lista dos indivíduos selecionados como elite.
        """
        print(f"\n--- Elitism Selection (elite_size={elite_size}) ---")
        
        # Ordena os indivíduos pela fitness e seleciona os top elite_size
        elite_indices = np.argsort(score)[-elite_size:]
        elite_individuals = [self.population[i] for i in elite_indices]
        
        # 
        # ========================================
        # PRINT RESULTS FOR TESTES USING PYTEST
        # ========================================
        print(f"Elite indices: {elite_indices}")
        print(f"Elite individuals: {elite_individuals}")
        
        return elite_individuals
    
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
        print(f"Scores: {selected_scores}")
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
        
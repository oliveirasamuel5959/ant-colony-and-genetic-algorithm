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
        
        Algoritmo ACO para TSP:
        1. Cada formiga constrói uma solução completa movendo-se de cidade em cidade
           baseada em feromônio e visibilidade
        2. As soluções são avaliadas (cálculo da distância total)
        3. A matriz de feromônios é atualizada (evaporação + depósito)
        4. O melhor caminho encontrado é rastreado
        
        Args:
            - None
        Returns:
            - None
        """
        all_paths = []
        distances = []
        
        print(f"\n--- ACO Iteration ---")
        
        # 1. Construção das soluções - cada formiga constrói um caminho completo
        for ant_id in range(self.num_ants):
            path = self._construct_ant_path()
            distance = self.calculate_total_distance(path)
            all_paths.append(path)
            distances.append(distance)
            
            # Atualiza o melhor caminho encontrado
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = path.copy()
                print(f"✓ Ant {ant_id}: Nova melhor solução encontrada! Distância: {self.best_distance:.2f}")
            else:
                print(f"  Ant {ant_id}: Distância: {distance:.2f}")
        
        # 2. Atualização dos feromônios (evaporação global + depósito de feromônio)
        self._update_pheromones(all_paths, distances)
        
        # 3. Rastreamento do histórico
        self.history.append(self.best_distance)
        print(f"Melhor distância global: {self.best_distance:.2f}")

    def _construct_ant_path(self) -> List[int]:
        """
        Constrói um caminho completo para uma formiga usando probabilidades baseadas em feromônio e visibilidade.
        
        Args:
            - None
        Returns:
            - List[int]: Caminho completo da formiga (ordem de visitação das cidades).
        """
        path = []
        visited = set()
        
        # Escolhe uma cidade inicial aleatória
        current_city = np.random.randint(0, self.num_cities)
        path.append(current_city)
        visited.add(current_city)
        
        # Constrói o caminho visitando todas as cidades
        while len(visited) < self.num_cities:
            next_city = self._select_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        return path

    def _select_next_city(self, current_city: int, visited: set) -> int:
        """
        Escolhe a próxima cidade baseada na equação de probabilidade do ACO.
        
        Probabilidade: P(i,j) = (τ(i,j)^α * η(i,j)^β) / Σ(τ(i,k)^α * η(i,k)^β)
        
        Onde:
        - τ(i,j): quantidade de feromônio no caminho (i,j)
        - η(i,j): visibilidade = 1 / distância(i,j)
        - α: importância relativa do feromônio
        - β: importância relativa da visibilidade
        - Σ: soma sobre todas as cidades não visitadas
        
        Args:
            - current_city (int): Cidade atual da formiga.
            - visited (set): Conjunto de cidades já visitadas pela formiga.
        Returns:
            - int: Próxima cidade selecionada.
        """
        # Calcula probabilidades para todas as cidades não visitadas
        probabilities = []
        unvisited_cities = []
        
        for city in range(self.num_cities):
            if city not in visited:
                unvisited_cities.append(city)
                
                # Calcula componentes
                pheromone = self.pheromone[current_city, city] ** self.alpha
                distance = self.dist_matrix[current_city, city]
                visibility = (1.0 / distance) if distance > 0 else 1.0
                visibility = visibility ** self.beta
                
                # Probabilidade: τ^α * η^β
                probability = pheromone * visibility
                probabilities.append(probability)
        
        # Normaliza probabilidades
        total_prob = sum(probabilities)
        if total_prob <= 0:
            # Fallback: seleção uniforme se todas as probabilidades forem zero
            return np.random.choice(unvisited_cities)
        
        probabilities = np.array(probabilities) / total_prob
        
        # Seleciona a próxima cidade com base nas probabilidades calculadas
        selected_city = np.random.choice(unvisited_cities, p=probabilities)
        
        return selected_city

    def _update_pheromones(self, all_paths: List[List[int]], distances: List[float]) -> None:
        """
        Atualiza a matriz de feromônios usando evaporação e depósito.
        
        Processo:
        1. Evaporação: τ(i,j) = (1 - ρ) * τ(i,j)
        2. Depósito: τ(i,j) += Σ(Δτ_k(i,j)) para cada formiga k
           onde Δτ_k = Q / L_k (Q é constante, L_k é distância do caminho da formiga k)
        
        Args:
            - all_paths (List[List[int]]): Lista de caminhos percorridos pelas formigas.
            - distances (List[float]): Lista de distâncias correspondentes a cada caminho.
        Returns:
            - None
        """
        # 1. Evaporação - reduz o feromônio em todos os caminhos
        self.pheromone *= (1 - self.rho)
        
        # Evita valores muito pequenos
        self.pheromone = np.maximum(self.pheromone, 1e-10)
        
        # 2. Depósito de feromônio - adiciona feromônio baseado na qualidade das soluções
        for path, distance in zip(all_paths, distances):
            # Quantidade de feromônio a depositar (inversamente proporcional à distância)
            # Formigas com caminhos menores depositam mais feromônio
            delta_pheromone = self.q / distance if distance > 0 else self.q
            
            # Deposita feromônio em cada aresta do caminho
            for i in range(len(path)):
                current_city = path[i]
                next_city = path[(i + 1) % len(path)]  # Próxima cidade (volta ao início ao fim)
                
                # Adiciona feromônio em ambas as direções (grafo não-dirigido)
                self.pheromone[current_city, next_city] += delta_pheromone
                self.pheromone[next_city, current_city] += delta_pheromone
        
    def get_pheromones(self) -> np.ndarray:
        """Retorna a matriz de feromônios para visualização.
        Args:
            - None
        Returns:
            - np.ndarray: Matriz de feromônios.
        """
        return self.pheromone

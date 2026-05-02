from solvers.genetic_algorithm import GeneticAlgorithm
from data.loader import load_from_json
import numpy as np

def test_genetic_algorithm():
    cities = load_from_json("data/cities.json")
    params = {"pop_size": 10, "mutation_rate": 0.05}
    
    ga_solver = GeneticAlgorithm(cities, params)
    
    print('\n' + 30*"-" + "\n" + "Genetic Algorithm Test" + "\n" + 30*"-")
    print(f"\n{ga_solver.__class__.__name__} initialized with {ga_solver.num_cities} cities and parameters: {ga_solver.params}")
    
    print("\n!--- Sample Teste de População Inicial ---")
    print(ga_solver.population[:10])
    
    # print("\n!--- Teste de Fitness ---")
    # ga_solver._test_fitness()
    
    # print("\n!--- Teste de Seleção ---")
    # ga_solver._test_selection()
    
    # print("\n!--- Teste de Crossover ---")
    # ga_solver._test_crossover()
    
    # print("\n!--- Teste de Mutação ---")
    # ga_solver._test_mutation()
    
    print("\n!--- Evolução do Algoritmo ---")
    ga_solver.evolve()
    print(f"Best path: {ga_solver.best_path}, Distance: {round(ga_solver.best_distance, 3)}")
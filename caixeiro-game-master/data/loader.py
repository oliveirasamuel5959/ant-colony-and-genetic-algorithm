import json
import csv
import numpy as np
from typing import List, Tuple
from pydantic import BaseModel, ValidationError

class City(BaseModel):
    id: int
    x: float
    y: float

def load_from_json(file_path: str) -> np.ndarray:
    """Carrega coordenadas de um arquivo JSON."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            cities = [City(**item) for item in data]
            return np.array([[c.x, c.y] for c in cities])
    except (json.JSONDecodeError, ValidationError, FileNotFoundError) as e:
        print(f"Erro ao carregar JSON: {e}")
        return np.array([])

def load_from_csv(file_path: str) -> np.ndarray:
    """Carrega coordenadas de um arquivo CSV (id, x, y)."""
    try:
        cities = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cities.append(City(id=int(row['id']), x=float(row['x']), y=float(row['y'])))
        return np.array([[c.x, c.y] for c in cities])
    except (ValidationError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"Erro ao carregar CSV: {e}")
        return np.array([])

def generate_random_cities(num_cities: int, width: int, height: int, save_path: str = None) -> np.ndarray:
    """Gera cidades aleatórias dentro de uma área e opcionalmente salva em JSON."""
    coords = np.random.rand(num_cities, 2) * [width, height]
    
    if save_path:
        cities = [City(id=i, x=float(coords[i, 0]), y=float(coords[i, 1])) for i in range(num_cities)]
        try:
            with open(save_path, 'w') as f:
                # model_dump() para Pydantic v2
                json.dump([c.model_dump() for c in cities], f, indent=4)
        except Exception as e:
            print(f"Erro ao salvar JSON: {e}")
            
    return coords

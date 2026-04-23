import math
import random
from typing import List

def get_city_name(index: int) -> str:
    name = ""
    while True:
        name = chr(65 + (index % 26)) + name
        index = (index // 26) - 1
        if index < 0:
            break
    return name

class City:
    def __init__(self, x: float, y: float, name: str = ""):
        self.x = x
        self.y = y
        self.name = name

    def distance_to(self, other: 'City') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)

def generate_cities(num_cities: int, start_x: int, start_y: int, end_x: int, end_y: int, padding: int = 50) -> List[City]:
    """Generates a random list of cities within a given bounding box."""
    cities = []
    for i in range(num_cities):
        x = random.randint(start_x + padding, end_x - padding)
        y = random.randint(start_y + padding, end_y - padding)
        cities.append(City(x, y, get_city_name(i)))
    return cities

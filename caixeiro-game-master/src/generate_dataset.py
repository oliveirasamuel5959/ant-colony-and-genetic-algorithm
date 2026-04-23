import sys
import os

# Adiciona o diretório atual ao path para encontrar os módulos locais
sys.path.append(os.getcwd())

from data.loader import generate_random_cities

def main():
    if len(sys.argv) < 2:
        print("Uso: uv run src/generate_dataset.py <quantidade_de_cidades>")
        return

    num_cities = int(sys.argv[1])
    # Gera cidades em uma escala HD padrão (1280x720) com margem
    generate_random_cities(num_cities, 1100, 600, save_path="data/cities.json")
    print(f"Sucesso: {num_cities} cidades geradas e salvas em 'data/cities.json'")

if __name__ == "__main__":
    main()

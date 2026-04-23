# TSP Educational Project - FEI

Este repositório contém um simulador interativo para o **Problema do Caixeiro Viajante (TSP)**, desenvolvido para fins educacionais na disciplina de **Introdução à Inteligência Artificial** no **Centro Universitário FEI**.

O objetivo do projeto é permitir que os alunos implementem e comparem duas meta-heurísticas clássicas: **Algoritmos Genéticos (GA)** e **Otimização por Colônia de Formigas (ACO)**.

## 🚀 Como Executar

O projeto utiliza o gerenciador de pacotes `uv` para garantir um ambiente consistente.

1. **Instale as dependências:**
   ```bash
   uv sync
   ```

2. **Execute o simulador:**
   ```bash
   uv run main.py
   ```

## 📂 Estrutura do Repositório

- `main.py`: Ponto de entrada da aplicação que inicializa o Pygame.

- `core/base_solver.py`: Contém a classe abstrata `TSPSolver`. Todos os algoritmos herdam desta classe.
- `solvers/`: **Local de trabalho do aluno.**
    - `genetic_algorithm.py`: Template para implementação do GA.
    - `aco_solver.py`: Template para implementação do ACO.
- `ui/`: Gerenciamento da interface gráfica (Pygame) e visualização em tempo real.
- `data/`: Contém o dataset de cidades (`cities.json`) e utilitários de carregamento.
- `src/`: Scripts auxiliares para geração de dados e lógica interna.

## 🛠️ Tarefa do Aluno

Os alunos devem completar as implementações nos arquivos dentro da pasta `solvers/`.

### 1. Algoritmo Genético (`solvers/genetic_algorithm.py`)
Você deve implementar os seguintes métodos:
- `evolve()`: Lógica principal de uma geração (seleção, cruzamento, mutação).
- `_crossover(parent1, parent2)`: Operador de cruzamento (ex: Order Crossover).
- `_mutate(individual)`: Operador de mutação (ex: Swap Mutation).

### 2. Colônia de Formigas (`solvers/aco_solver.py`)
Você deve implementar os seguintes métodos:
- `evolve()`: Lógica de uma iteração (movimentação de todas as formigas e atualização de feromônios).
- `_select_next_city(current_city, visited)`: Regra de transição probabilística baseada em feromônios e visibilidade.
- `_update_pheromones(all_paths, distances)`: Depósito de feromônios nos caminhos encontrados e evaporação.

## 🎮 Controles da Interface

Ao executar o simulador, utilize as seguintes teclas:

- **Modos de Visualização:**
    - `1`: Menu de Configuração (Setup).
    - `2`: Visualização do **Algoritmo Genético**.
    - `3`: Visualização da **Colônia de Formigas**.
    - `4`: Comparação **Lado a Lado** (GA vs ACO).

- **Simulação:**
    - `ESPAÇO`: Iniciar / Pausar a evolução.
    - `R`: Resetar a simulação atual (reinicia os solvers com parâmetros iniciais).
    - `G`: Gerar um novo dataset aleatório (salva em `data/cities.json`).

## 📊 Visualização
O simulador exibe:
- O melhor caminho encontrado em tempo real.
- A intensidade dos feromônios (no caso do ACO).
- Um gráfico de convergência (Distância vs. Iteração).
- Estatísticas de performance comparativas.

---
**Professor:** Rafael Gomes Alves  
**Instituição:** Centro Universitário FEI  
**Disciplina:** Introdução à Inteligência Artificial

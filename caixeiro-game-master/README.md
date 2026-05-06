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

## 🔬 Detalhes dos Algoritmos Implementados

### Algoritmo Genético (`solvers/genetic_algorithm.py`)
O GA implementa uma abordagem evolutiva para o TSP:
- **Evolução**: A cada geração, seleciona os melhores indivíduos, aplica cruzamento (Order Crossover) e mutação (Swap Mutation) para gerar novos descendentes.
- **Seleção**: Escolhe os indivíduos com menor distância total do caminho.
- **Cruzamento e Mutação**: Cria diversidade genética mantendo características dos melhores candidatos.

### Colônia de Formigas (`solvers/aco_solver.py`)
O ACO simula o comportamento de formigas reais:
- **Movimentação**: Cada formiga constrói um caminho usando regra de transição probabilística baseada em feromônios (α) e visibilidade (β).
- **Atualização de Feromônios**: Após cada iteração, formigas depositam feromônios nos caminhos percorridos, com evaporação de feromônios antigos.
- **Convergência**: O algoritmo tende a convergir para boas soluções à medida que feromônios se acumulam em caminhos promissores.

## 📈 Geração de Gráficos e Análise

### Como Usar os Gráficos
Ao executar uma simulação (GA, ACO ou Comparação), os gráficos de convergência são automaticamente salvos ao pressionar `1` (Setup) após terminar as iterações/gerações. Esses gráficos mostram a evolução da melhor distância encontrada ao longo do tempo.

### Tipos de Gráficos Gerados
- **Gráfico de Convergência (GA)**: Visualiza como o Algoritmo Genético melhora a solução a cada geração (eixo X = gerações, eixo Y = melhor distância).
- **Gráfico de Convergência (ACO)**: Visualiza como a Colônia de Formigas melhora a solução a cada iteração (eixo X = iterações, eixo Y = melhor distância).
- **Gráfico Comparativo**: Exibe ambos os algoritmos em um único gráfico, permitindo comparar suas velocidades de convergência e qualidade final das soluções.

### Estrutura de Logs
Quando você executa os modos **GA** (`2`), **ACO** (`3`) ou **Comparação** (`4`), um diretório `logs/` é automaticamente criado na raiz do projeto com a seguinte estrutura:

```
logs/
├── ago/              # Gráficos do Algoritmo Genético (GA)
│   └── *.png        # Convergence plots com diferentes parâmetros
├── aco/              # Gráficos da Colônia de Formigas (ACO)
│   └── *.png        # Convergence plots com diferentes parâmetros
└── *.png            # Gráficos comparativos (GA vs ACO)
```

Os arquivos de imagem são nomeados dinamicamente com base nos parâmetros utilizados, facilitando a comparação entre diferentes execuções.

---
**Professor:** Rafael Gomes Alves  
**Instituição:** Centro Universitário FEI  
**Disciplina:** Introdução à Inteligência Artificial

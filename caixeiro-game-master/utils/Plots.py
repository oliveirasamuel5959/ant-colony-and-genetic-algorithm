import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def save_history_and_plots(history, algo=None, title="Convergência do Algoritmo", filename="convergence_plot.png"):
    plt.figure(figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
    
    if algo == 'ago':
        plt.plot(history["iterations"], history["ago_best_distance"], linestyle='-', color='r')
        plt.xlabel("gerações")
        plt.ylim(2000, max(history["ago_best_distance"]) * 1.1)
    elif algo == 'aco':
        plt.plot(history["iterations"], history["aco_best_distance"], linestyle='-', color='b')
        plt.xlabel("iterações")
        plt.ylim(2000, max(history["aco_best_distance"]) * 1.1)
    elif algo == 'comparison':
        plt.plot(history["iterations"], history["ago_best_distance"], linestyle='-', color='r', label='Genetic Algorithm')
        plt.plot(history["iterations"], history["aco_best_distance"], linestyle='-', color='b', label='Ant Colony Optimization')
        plt.xlabel("gerações/iterações")
        plt.ylim(2000, max(history["ago_best_distance"]) * 1.1)
        plt.legend()
        
    plt.title(title)
    plt.xlim(1, max(history["iterations"]) * 1.1)
    plt.ylabel("Melhor Distância")
    plt.grid()
    plt.savefig(filename)
    plt.show()
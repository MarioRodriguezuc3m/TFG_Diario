import matplotlib.pyplot as plt  # Asegúrate de tener matplotlib instalado
from Graph import Graph
from Ant import Ant
from typing import Dict
class ACO:
    def __init__(self, graph: Graph,  consultas_orden: Dict[str, int] , consultas_duration: Dict[str,int],n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
        self.graph = graph
        self.consultas_orden = consultas_orden
        self.consultas_duration = consultas_duration
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_solution = None
        self.best_cost = float('inf')

    def run(self):
        for _ in range(self.iterations):
            ants = [Ant(self.graph, self.consultas_orden, self.consultas_duration,self.alpha, self.beta) for _ in range(self.n_ants)]
            for ant in ants:
                while True:
                    next_node = ant.choose_next_node()
                    if next_node is None or ant.valid_solution:
                        break
                    ant.move(next_node)
                if ant.valid_solution:
                    ant.total_cost = ant.calcular_coste(ant.visited)
                    if ant.total_cost < self.best_cost:
                        self.best_cost = ant.total_cost
                        self.best_solution = ant.visited.copy()
                else:
                    ant.total_cost = float('inf')
            
            self.graph.update_pheromone(ants, self.rho, self.Q)
        
        return self.best_solution, self.best_cost

    def plot_convergence(self):
        plt.plot(self.best_distances)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.title('Convergencia del ACO')
        plt.show()
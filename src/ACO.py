import matplotlib.pyplot as plt  # Asegúrate de tener matplotlib instalado
from Graph import Graph
from Ant import Ant
from typing import Dict,List
import time  # Importa el módulo time
import os

class ACO:
    def __init__(self, graph: Graph,  fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str], n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
        self.graph = graph
        self.fases_orden = fases_orden  
        self.fases_duration = fases_duration  
        self.pacientes = pacientes
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_solution = None
        self.total_costs = []
        self.best_cost = float('inf')
        self.execution_time = None

    def run(self):
        start_time = time.time()  # Marca el tiempo de inicio
        
        for _ in range(self.iterations):
            ants = [Ant(self.graph, self.fases_orden, self.fases_duration, self.pacientes, self.alpha, self.beta) for _ in range(self.n_ants)]
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
            self.total_costs.append(self.best_cost)
            self.graph.update_pheromone(ants, self.rho, self.Q)
        
        end_time = time.time()  # Marca el tiempo de fin
        self.execution_time = end_time - start_time  # Calcula el tiempo total de ejecución

        return self.best_solution, self.best_cost


    def plot_convergence(self):
        plt.plot(self.total_costs)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.title('Convergencia del ACO')
        
        # Crear directorio si no existe
        os.makedirs("/app/plots", exist_ok=True)
        
        # Guardar la imagen
        plt.savefig("/app/plots/convergencia.png")
        plt.close()  # Limpiar la figura

    def get_execution_time(self):
        """Devuelve el tiempo de ejecución total en segundos."""
        return self.execution_time


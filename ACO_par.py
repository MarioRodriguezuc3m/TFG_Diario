import matplotlib.pyplot as plt  # Asegúrate de tener matplotlib instalado
from Graph import Graph
from Ant import Ant
from typing import Dict,List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

class ACO_parallel:
    def __init__(self, graph: Graph, fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str], n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0, num_parallel_ants: int = 5):
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
        self.num_parallel_ants = num_parallel_ants
        self.best_solution = None
        self.total_costs = []
        self.best_cost = float('inf')
        self.execution_time = 0
        self.lock = threading.Lock()  # Lock para proteger el acceso a best_cost

    def run(self):
        start_time = time.time()  # Marca el tiempo de inicio
        
        # Ejecutar el ciclo principal para las iteraciones
        for _ in range(self.iterations):
            ants = [Ant(self.graph, self.fases_orden, self.fases_duration, self.pacientes, self.alpha, self.beta) for _ in range(self.n_ants)]
            
            # Ejecutar las hormigas en paralelo con un máximo de hilos en ejecución
            with ThreadPoolExecutor(max_workers=self.num_parallel_ants) as executor:
                futures = [executor.submit(self.run_ant, ant) for ant in ants]
                completed_ants = []  # Lista para almacenar las hormigas que terminaron

                for future in as_completed(futures):
                    result = future.result()
                    ant = result['ant']
                    total_cost = result['cost']
                    
                    # Actualizar el mejor costo y solución, asegurando que el acceso sea seguro
                    with self.lock:
                        if total_cost < self.best_cost:
                            self.best_cost = total_cost
                            self.best_solution = ant.visited.copy()
                    
                    # Almacenar la hormiga y su costo calculado
                    completed_ants.append(ant)
            
            self.total_costs.append(self.best_cost)
            
            # Actualizar feromonas con las hormigas completas
            self.graph.update_pheromone(completed_ants, self.rho, self.Q)
        
        end_time = time.time()  # Marca el tiempo de fin
        self.execution_time = end_time - start_time  # Calcula el tiempo total de ejecución
        
        return self.best_solution, self.best_cost

    def run_ant(self, ant: Ant):
        # El ciclo de la hormiga: elige el siguiente nodo y se mueve por el grafo
        while True:
            next_node = ant.choose_next_node()
            if next_node is None or ant.valid_solution:
                break
            ant.move(next_node)
        
        # Calcular el costo si la solución es válida
        if ant.valid_solution:
            ant.total_cost = ant.calcular_coste(ant.visited)
        else:
            ant.total_cost = float('inf')
    
        return {'ant': ant, 'cost': ant.total_cost}
    
    def plot_convergence(self):
        plt.plot(self.total_costs)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Costo')
        plt.title('Convergencia del ACO')
        plt.show()

    def get_execution_time(self):
        """Devuelve el tiempo de ejecución total en segundos."""
        return self.execution_time

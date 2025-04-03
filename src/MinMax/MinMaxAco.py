from Standard.ACO import ACO
from MinMax.MinMaxGraph import MinMaxGraph 
from typing import List, Dict
from utils.Ant import Ant
import matplotlib.pyplot as plt 
import time
import os

class MinMaxACO(ACO): 
    def __init__(self, 
                 graph: MinMaxGraph,  # Ahora usa MinMaxGraph específicamente
                 fases_orden: Dict[str, int], 
                 fases_duration: Dict[str, int], 
                 pacientes: List[str],
                 medicos: List[str],
                 consultas: List[str],
                 horas: List[str], 
                 n_ants: int = 10, 
                 iterations: int = 100,
                 alpha: float = 1.0, 
                 beta: float = 3.0, 
                 rho: float = 0.1, 
                 Q: float = 1.0):
        
        super().__init__(graph, fases_orden, fases_duration, pacientes, medicos, 
                       consultas, horas, n_ants, iterations, alpha, beta, rho, Q)
        
        # Parámetros específicos de MinMax
        self.graph: MinMaxGraph = graph  # Type hint específico

    def run(self):
        start_time = time.time()
        
        for _ in range(self.iterations):
            ants = [Ant(self.graph, self.fases_orden, self.fases_duration, 
                       self.pacientes, self.alpha, self.beta) 
                   for _ in range(self.n_ants)]
            
            iteration_best = None
            iteration_best_cost = float('inf')
            
            # Construcción de soluciones
            for ant in ants:
                while True:
                    next_node = ant.choose_next_node()
                    if next_node is None or ant.valid_solution:
                        break
                    ant.move(next_node)
                
                if ant.valid_solution:
                    ant.total_cost = self.calcular_coste(ant.visited)
                    if ant.total_cost < iteration_best_cost:
                        iteration_best = ant
                        iteration_best_cost = ant.total_cost
            
            # Búsqueda local y actualización global
            if iteration_best:
                improved = self.local_search(iteration_best.visited)
                improved_cost = self.calcular_coste(improved)
                
                if improved_cost < self.best_cost:
                    self.best_cost = improved_cost
                    self.best_solution = improved.copy()
            
            # Actualización de feromonas (diferente al ACO original)
            self.graph.update_pheromone(
                best_ant=iteration_best,
                rho=self.rho,
                Q=self.Q
            )
            
            self.total_costs.append(self.best_cost)
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_cost
    
    def plot_convergence(self):
        plt.plot(self.total_costs)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.title('Convergencia del ACO')
        
        # Crear directorio si no existe
        os.makedirs("/app/plots", exist_ok=True)
        
        # Guardar la imagen
        plt.savefig("/app/plots/convergenciaMinMax.png")
        plt.close()  # Limpiar la figura
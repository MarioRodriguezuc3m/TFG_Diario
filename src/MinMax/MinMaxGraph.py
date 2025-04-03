from utils.Ant import Ant
from Standard.Graph import Graph
from typing import List, Dict, Tuple
class MinMaxGraph(Graph):
    def __init__(self, 
                 nodes: List[Tuple], 
                 edges: Dict[Tuple, List[Tuple]], 
                 pheromone_max: float = 100.0,
                 pheromone_min: float = 0.1,
                 initial_pheromone: float = None):
        
        # Inicializar con feromona máxima por defecto
        initial_pheromone = pheromone_max if initial_pheromone is None else initial_pheromone
        super().__init__(nodes, edges, initial_pheromone)
        
        # Límites de feromonas
        self.pheromone_max = pheromone_max
        self.pheromone_min = pheromone_min
        
        # Asegurar que todas las feromonas iniciales están en los límites
        for edge in self.pheromone:
            self.pheromone[edge] = pheromone_max

    def update_pheromone(self, best_ant: Ant, rho: float, Q: float):
        # 1. Evaporación global con límite inferior
        for edge in self.pheromone:
            evaporated = self.pheromone[edge] * (1 - rho)
            self.pheromone[edge] = max(evaporated, self.pheromone_min)
        
        # 2. Actualización solo de la mejor hormiga si existe y es válida
        if best_ant and best_ant.valid_solution and best_ant.total_cost > 0:
            delta = Q / best_ant.total_cost
            path = best_ant.visited
            
            # 3. Aplicar refuerzo a la mejor ruta
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                current_pheromone = self.pheromone.get(edge, self.pheromone_min)
                new_pheromone = current_pheromone + delta
                self.pheromone[edge] = min(new_pheromone, self.pheromone_max)
        
        # 4. Aplicar límites superiores a todas las aristas
        for edge in self.pheromone:
            self.pheromone[edge] = min(self.pheromone[edge], self.pheromone_max)
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from Ant import Ant

class Graph:
    def __init__(self, nodes: List[Tuple], edges: Dict[Tuple, List[Tuple]], initial_pheromone: float = 1.0):
        self.nodes = nodes
        self.edges = edges
        self.initial_pheromone = initial_pheromone
        self.pheromone = defaultdict(float)
        
        # Inicializar feromonas usando tuplas como claves
        for node in edges:
            for neighbor in edges[node]:
                self.pheromone[(node, neighbor)] = initial_pheromone

    def get_pheromone(self, node1: Tuple, node2: Tuple) -> float:
        # Acceso directo con tupla como clave
        return self.pheromone.get((node1, node2), self.initial_pheromone)

    def update_pheromone(self, ants: List[Ant], rho: float, Q: float):
        # 1. Evaporación global
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - rho)
        
        # 2. Actualización basada en hormigas
        for ant in ants:
            if not ant.visited:
                continue
                
            # Calcular delta según validez de la solución
            if ant.valid_solution and ant.total_cost > 0:
                delta = Q / ant.total_cost
            else:
                delta = -Q * 0.1  # Penalización para soluciones inválidas
            
            # Aplicar a todas las aristas del camino
            for i in range(len(ant.visited) - 1):
                edge = (ant.visited[i], ant.visited[i+1])
                if edge in self.pheromone:
                    self.pheromone[edge] += delta
                else:
                    # Opcional: Inicializar feromona para nuevas conexiones
                    self.pheromone[edge] = self.initial_pheromone + delta
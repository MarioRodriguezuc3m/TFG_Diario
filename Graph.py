from typing import List, Dict, Tuple, Set
from collections import defaultdict
from Ant import Ant
import datetime

class Graph:
    def __init__(self, nodes: List[Tuple], edges: Dict[Tuple, List[Tuple]],pacientes: List[str] = [],initial_pheromone: float = 1.0):
        self.pacientes = pacientes
        self.nodes = nodes
        self.edges = edges
        self.pheromone = defaultdict(dict)
        
        # Inicializar feromonas
        for node in self.nodes:
            for neighbor in self.edges[node]:
                self.pheromone[node][neighbor] = initial_pheromone

    def get_pheromone(self, node1: Tuple, node2: Tuple) -> float:
        return self.pheromone.get(node1, {}).get(node2, 0.0)

    def update_pheromone(self, ants: List['Ant'], rho: float, Q: float):
        # Evaporación
        for node in self.nodes:
            for neighbor in self.edges[node]:
                self.pheromone[node][neighbor] *= (1 - rho)
        
        # Deposición de feromona
        for ant in ants:
            if not ant.valid_solution:
                continue
                
            delta_pheromone = Q / ant.total_cost if ant.total_cost != 0 else 0
            for i in range(len(ant.visited)-1):
                current = ant.visited[i]
                next_node = ant.visited[i+1]
                self.pheromone[current][next_node] += delta_pheromone
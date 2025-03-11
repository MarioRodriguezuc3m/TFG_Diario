from typing import List, Dict, Tuple, Set
from collections import defaultdict
from Ant import Ant
import datetime

class Graph:
    def __init__(self, nodes: List[Tuple], edges: Dict[Tuple, List[Tuple]], consultas_orden: Dict[str, int],pacientes: List[str] = [],initial_pheromone: float = 1.0):
        self.pacientes = pacientes
        self.nodes = nodes
        self.edges = edges
        self.consultas_orden = consultas_orden
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
                
    def calcular_tiempo_espera(self, asignaciones: List[Tuple]) -> float:
        tiempos_pacientes = defaultdict(list)
        duracion_consultas = {"ConsultaA": 60, "ConsultaB": 60}  # Duración en minutos
        
        # Registrar tiempos de inicio y fin por paciente
        for asignacion in asignaciones:
            paciente, consulta, hora_str, _ = asignacion
            hora_inicio = datetime.datetime.strptime(hora_str, "%H:%M").time()
            inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
            fin_min = inicio_min + duracion_consultas[consulta]
            tiempos_pacientes[paciente].append((self.consultas_orden[consulta], inicio_min, fin_min))
        
        total_espera = 0
        for paciente, tiempos in tiempos_pacientes.items():
            # Ordenar por orden de consultas
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])
            
            # Calcular tiempo entre fin de consulta anterior e inicio de la siguiente
            for i in range(1, len(tiempos_ordenados)):
                fin_anterior = tiempos_ordenados[i-1][2]
                inicio_actual = tiempos_ordenados[i][1]
                total_espera += max(inicio_actual - fin_anterior, 0)  # Solo tiempo positivo
                
        return total_espera
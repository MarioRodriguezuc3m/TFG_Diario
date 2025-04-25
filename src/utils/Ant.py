import random
from utils.generate_graph_components import *
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Standard.Graph import Graph

class Ant:
    def __init__(self, graph: "Graph", fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str], alpha: float = 1.0, beta: float = 1.0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.visited: List[Tuple] = []
        self.fases_orden = fases_orden  
        self.fases_duration = fases_duration  
        self.pacientes = pacientes
        self.pacientes_progreso = defaultdict(dict, {paciente: {} for paciente in self.pacientes}) # Se utiliza un diccionario para mantener el progreso de cada paciente
        self.current_node: Tuple = None
        self.total_cost: float = 0.0
        self.valid_solution = False

    def choose_next_node(self) -> Tuple:
        if self.current_node is None:
            # Se eligen aleatoriamente los nodos iniciales válidos (solo nodos de la primera fase)
            valid_initial_nodes = [
                node for node in self.graph.nodes
                if self.fases_orden[node[4]] == 1  # Se seleccionan únicamente nodos de Fase1 (orden 1)
            ]
            return random.choice(valid_initial_nodes) if valid_initial_nodes else None
        else:
            candidates = self.graph.edges.get(self.current_node, [])
            # Filtrar candidatos para evitar pacientes que ya han completado todas las fases
            candidates = [
                node for node in candidates
                if len(self.pacientes_progreso[node[0]]) < len(self.fases_orden)
            ]
            if not candidates:
                return None

            candidate_list = []
            probabilities = []
            total = 0.0

            for node in candidates:
                heuristic = self.calcular_heuristica(node)
                # Se obtiene el valor de la feromona para la transición actual
                pheromone = self.graph.get_pheromone(self.current_node, node)
                candidate_weight = (pheromone ** self.alpha) * (heuristic ** self.beta)
                candidate_list.append(node)
                probabilities.append(candidate_weight)
                total += candidate_weight

            if total == 0:
                return random.choice(candidate_list)

            return random.choices(candidate_list, weights=[p / total for p in probabilities], k=1)[0]


    def calcular_heuristica(self, node: Tuple) -> float:
        """
        Función para calcular el valor heuristico, comparando el current_node, con el nodo(node) al que se quiere transicionar.
        """
        paciente, consulta, hora, medico, fase = node
        
        # Puntuación base
        score = 10.0
        
        hora_parts = hora.split(':')
        node_mins = int(hora_parts[0]) * 60 + int(hora_parts[1])
        
        if self.current_node:
            current_hora_parts = self.current_node[2].split(':')
            current_mins = int(current_hora_parts[0]) * 60 + int(current_hora_parts[1])
            duracion_actual = self.fases_duration[self.current_node[4]]
            
            # Si el paciente del siguiente nodo es el mismo que el del nodo actual:
            if paciente == self.current_node[0]:
                # Se verifica si la siguiente fase comienza después de que se complete la actual
                if node_mins < current_mins + duracion_actual:
                    score -= 20.0  # Se aplica una fuerte penalización por superposición de tiempo
                else:
                    # Se calcula el tiempo de espera entre el nodo actual y el siguiente
                    tiempo_espera = node_mins - (current_mins + duracion_actual)
                    if tiempo_espera <= 60:
                        score += 10.0  # Continuidad de tiempo óptima
                    elif tiempo_espera <= 120:
                        score += 6.0   # Tiempo de espera razonable
                    elif tiempo_espera > 120:  # Más de 3 horas
                        score -= min(10.0, (tiempo_espera - 180) / 60.0 * 2.0)  # Penalización
        

        node_end_mins = node_mins + self.fases_duration[fase]
        
        # Variables para almacenar el uso de recursos (equilibrio simplificado)
        medico_count = 0
        consulta_count = 0
        
        # Para los nodos visitados por la hormiga, se evalua si no hay conflictos de recursos, con el nodo al que se va a transicionar
        for v_node in self.visited:
            v_paciente, v_consulta, v_hora, v_medico, v_fase = v_node
            
            v_hora_parts = v_hora.split(':')
            v_mins = int(v_hora_parts[0]) * 60 + int(v_hora_parts[1])
            v_end_mins = v_mins + self.fases_duration[v_fase]
            
            if node_mins < v_end_mins and v_mins < node_end_mins:
                if v_medico == medico:
                    score -= 15.0  # Conflicto de médico
                if v_consulta == consulta:
                    score -= 15.0  # Conflicto de sala de consulta
            
            # Se cuenta el uso de recursos
            if v_medico == medico:
                medico_count += 1
            if v_consulta == consulta:
                consulta_count += 1
        
        # Se favorece el uso de recursos no  utilizados
        if medico_count == 0:
            score += 2.0 
        if consulta_count == 0:
            score += 2.0
        
        # Se asegura una heurística positiva
        return max(0.1, score)

    def move(self, node: Tuple):
        self.current_node = node
        self.visited.append(node)
        paciente, _, hora, _ , fase = node = node
        self.pacientes_progreso[paciente][fase] = hora
        
        # Se verifica que la solución sea completa y que el orden sea correcto
        self.valid_solution = all(
            self._fases_en_orden_correcto(fases) 
            for fases in self.pacientes_progreso.values()
        )

    def _fases_en_orden_correcto(self, fases_paciente: Dict) -> bool:
        """Verifica si las fases de un paciente están en el orden correcto."""
        fases = list(fases_paciente.keys())
        orden = [self.fases_orden[f] for f in fases]
        return orden == sorted(orden) and len(orden) == len(self.fases_orden)
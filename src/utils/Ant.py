import random
from utils.generate_graph_components import *
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Standard.Graph import Graph  # Se usa solo para type hint

class Ant:
    def __init__(self, graph: "Graph", fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str], alpha: float = 1.0, beta: float = 1.0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.visited: List[Tuple] = []
        self.fases_orden = fases_orden  
        self.fases_duration = fases_duration  
        self.pacientes = pacientes
        self.pacientes_progreso = defaultdict(dict, {paciente: {} for paciente in self.pacientes}) #Diccionario para mantener el progreso de cada paciente
        self.current_node: Tuple = None
        self.total_cost: float = 0.0
        self.valid_solution = False

    def choose_next_node(self) -> Tuple:
        if self.current_node is None:
            # Elegir aleatoriamente el primer nodo (solo nodos iniciales válidos)
            valid_initial_nodes = [
                node for node in self.graph.nodes
                if self.fases_orden[node[4]] == 1  # Solo Fase1 (orden 1)
            ]
            return random.choice(valid_initial_nodes) if valid_initial_nodes else None
        else:
            candidates = self.graph.edges.get(self.current_node, [])
            if not candidates:
                return None

            candidate_list = []
            probabilities = []
            total = 0.0

            for node in candidates:
                heuristic = self.calcular_heuristica(node)
                # Obtener el valor de la feromona para la transición actual
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
        Simplified heuristic calculation with better time efficiency.
        """
        paciente, consulta, hora, medico, fase = node
        
        # Base score
        score = 10.0
        
        # Fast time conversion - avoid datetime parsing
        hora_parts = hora.split(':')
        node_mins = int(hora_parts[0]) * 60 + int(hora_parts[1])
        
        # Process current node time comparison if not first node
        if self.current_node:
            current_hora_parts = self.current_node[2].split(':')
            current_mins = int(current_hora_parts[0]) * 60 + int(current_hora_parts[1])
            duracion_actual = self.fases_duration[self.current_node[4]]
            
            # Time factor - simplified calculation
            if paciente == self.current_node[0]:
                # Check if next phase starts after current one completes
                if node_mins < current_mins + duracion_actual:
                    score -= 20.0  # Strong penalty for time overlap
                else:
                    # Optimize time gap evaluation
                    tiempo_espera = node_mins - (current_mins + duracion_actual)
                    if tiempo_espera <= 60:
                        score += 10.0  # Optimal time continuity
                    elif tiempo_espera <= 120:
                        score += 6.0   # Reasonable wait time
                    elif tiempo_espera > 120:  # More than 3 hours
                        score -= min(10.0, (tiempo_espera - 180) / 60.0 * 2.0)  # Limited penalty
        
        # Track conflicts with a single pass through visited nodes
        # Using node time ranges to check for conflicts
        node_end_mins = node_mins + self.fases_duration[fase]
        
        # Counter for resource usage (simplified balancing)
        medico_count = 0
        consulta_count = 0
        
        for v_node in self.visited:
            v_paciente, v_consulta, v_hora, v_medico, v_fase = v_node
            
            # Fast time conversion
            v_hora_parts = v_hora.split(':')
            v_mins = int(v_hora_parts[0]) * 60 + int(v_hora_parts[1])
            v_end_mins = v_mins + self.fases_duration[v_fase]
            
            # Check for time overlap
            if node_mins < v_end_mins and v_mins < node_end_mins:
                # Resource conflicts
                if v_medico == medico:
                    score -= 15.0  # Doctor conflict
                if v_consulta == consulta:
                    score -= 15.0  # Consultation room conflict
            
            # Count resource usage for balancing
            if v_medico == medico:
                medico_count += 1
            if v_consulta == consulta:
                consulta_count += 1
        
        # Simple balancing boost based on current counts
        if medico_count == 0:
            score += 2.0  # Boost for unused doctor
        if consulta_count == 0:
            score += 2.0  # Boost for unused consultation room
        
        # Ensure positive heuristic
        return max(0.1, score)

    def move(self, node: Tuple):
        self.current_node = node
        self.visited.append(node)
        paciente, _, hora, _ , fase = node = node
        self.pacientes_progreso[paciente][fase] = hora
        
        # Verificar solución completa y orden correcto
        self.valid_solution = all(
            self._fases_en_orden_correcto(fases) 
            for fases in self.pacientes_progreso.values()
        )

    def _fases_en_orden_correcto(self, fases_paciente: Dict) -> bool:
        """Verifica si las fases de un paciente están en el orden correcto."""
        fases = list(fases_paciente.keys())
        orden = [self.fases_orden[f] for f in fases]
        return orden == sorted(orden) and len(orden) == len(self.fases_orden)
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
        """Calcula la heurística para un nodo candidato basado en penalizaciones."""
        penalty = 0.0
        current_paciente = self.current_node[0]
        paciente, _, hora, _, fase = node
        
        # Determinar si el paciente actual completó todas sus fases
        current_fases_completadas = self.pacientes_progreso[current_paciente].keys()
        current_fases_requeridas = len(self.fases_orden)
        paciente_actual_completo = len(current_fases_completadas) == current_fases_requeridas

        # 1. Penalizar cambiar de paciente si el actual no completó fases
        if not paciente_actual_completo and paciente != current_paciente:
            penalty += 1.0

        # 2. Penalizar nuevo paciente que no inicia con fase 1
        if paciente != current_paciente and self.fases_orden[fase] != 1:
            penalty += 1.0

        # 3. Verificar secuencia de fases para el mismo paciente
        if paciente == current_paciente:
            # Orden de la fase actual vs siguiente
            orden_actual = self.fases_orden[self.current_node[4]]
            orden_siguiente = self.fases_orden[fase]
            
            if orden_siguiente != orden_actual + 1:
                penalty += 1.0
            else:
                # Validar tiempo posterior
                hora_actual = datetime.datetime.strptime(self.current_node[2], "%H:%M")
                hora_siguiente = datetime.datetime.strptime(hora, "%H:%M")
                if hora_siguiente <= hora_actual:
                    penalty += 1.0

        # 4. Conflictos de médico en misma hora
        for visited_node in self.visited:
            if visited_node[3] == node[3] and visited_node[2] == node[2]:
                penalty += 1.0

        # 5. Conflictos de fase en misma hora
        for visited_node in self.visited:
            if visited_node[4] == node[4] and visited_node[2] == node[2]:
                penalty += 1.0

        return 1.0 / (1.0 + penalty)

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
import random
from typing import List, Dict, Tuple, Set, TYPE_CHECKING
from collections import defaultdict
import datetime

if TYPE_CHECKING:
    from Standard.Graph import Graph # Asumiendo que Graph está en Standard.Graph
    # from utils.generate_graph_components import ... # No es necesario aquí si Graph y ACO manejan esto

class Ant:
    def __init__(self, graph: "Graph", paciente_to_estudio_info: Dict[str, Dict], pacientes: List[str], alpha: float = 1.0, beta: float = 1.0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.visited: List[Tuple] = []
        
        # Información específica del estudio por paciente
        self.paciente_to_estudio_info = paciente_to_estudio_info
        
        self.pacientes = pacientes # Lista de todos los pacientes que deben ser programados
        self.pacientes_progreso = defaultdict(dict) # {paciente: {fase: hora}}
        
        self.current_node: Tuple = None
        self.total_cost: float = 0.0
        self.valid_solution = False

    def choose_next_node(self) -> Tuple:
        if self.current_node is None:
            # Se eligen aleatoriamente los nodos iniciales válidos (solo nodos de la primera fase de su respectivo estudio)
            valid_initial_nodes = [
                node for node in self.graph.nodes
                if node[0] in self.paciente_to_estudio_info and \
                   self.paciente_to_estudio_info[node[0]]["orden_fases"].get(node[4]) == 1
            ]
            return random.choice(valid_initial_nodes) if valid_initial_nodes else None
        else:
            candidates = self.graph.edges.get(self.current_node, [])
            
            # Filtrar candidatos para:
            # 1. Evitar pacientes que ya han completado todas las fases de su estudio.
            # 2. Evitar nodos que ya han sido visitados (misma tupla exacta).
            # 3. Evitar programar una fase para un paciente si esa fase ya fue programada para él.
            filtered_candidates = []
            for node in candidates:
                paciente_candidato = node[0]
                fase_candidata = node[4]

                if paciente_candidato not in self.paciente_to_estudio_info:
                    continue # Paciente no tiene info de estudio (no debería pasar)

                info_estudio_candidato = self.paciente_to_estudio_info[paciente_candidato]
                
                # Condición 1
                if len(self.pacientes_progreso[paciente_candidato]) >= len(info_estudio_candidato["orden_fases"]):
                    continue
                
                # Condición 2 (Evitar revisitar el mismo estado exacto)
                if node in self.visited: # Podría ser demasiado restrictivo si se permite volver sobre pasos para optimizar
                    pass # Por ahora, permitimos si las otras condiciones se cumplen.ACO suele permitir esto.

                # Condición 3 (Evitar programar la misma fase dos veces para el mismo paciente)
                if fase_candidata in self.pacientes_progreso[paciente_candidato]:
                    continue
                
                filtered_candidates.append(node)

            if not filtered_candidates:
                return None

            candidate_list = []
            probabilities = []
            total_prob_weight = 0.0

            for node in filtered_candidates:
                heuristic = self.calcular_heuristica(node)
                pheromone = self.graph.get_pheromone(self.current_node, node)
                candidate_weight = (pheromone ** self.alpha) * (heuristic ** self.beta)
                
                candidate_list.append(node)
                probabilities.append(candidate_weight)
                total_prob_weight += candidate_weight

            # Normalizar probabilidades
            normalized_probabilities = [p / total_prob_weight for p in probabilities]
            return random.choices(candidate_list, weights=normalized_probabilities, k=1)[0]


    def calcular_heuristica(self, node_to_evaluate: Tuple) -> float:
        paciente, consulta, hora_str, medico, fase = node_to_evaluate
        
        # Puntuación base
        score = 10.0
        
        if paciente not in self.paciente_to_estudio_info:
             return 0.1 # Paciente no reconocido, muy baja heurística

        info_estudio_paciente_eval = self.paciente_to_estudio_info[paciente]
        
        hora_parts = hora_str.split(':')
        node_mins = int(hora_parts[0]) * 60 + int(hora_parts[1])
        
        if self.current_node:
            curr_paciente, _, curr_hora_str, _, curr_fase = self.current_node
            info_estudio_paciente_curr = self.paciente_to_estudio_info[curr_paciente]

            current_hora_parts = curr_hora_str.split(':')
            current_mins = int(current_hora_parts[0]) * 60 + int(current_hora_parts[1])
            duracion_fase_actual = info_estudio_paciente_curr["fases_duration"][curr_fase]
            
            if paciente == curr_paciente: # Mismo paciente
                # Verificar si la siguiente fase propuesta comienza después de que la actual termine
                if node_mins < current_mins + duracion_fase_actual:
                    score -= 20.0  # Fuerte penalización por superposición de tiempo para el mismo paciente
                else:
                    tiempo_espera = node_mins - (current_mins + duracion_fase_actual)
                    if tiempo_espera <= 60: score += 10.0
                    elif tiempo_espera <= 120: score += 6.0
                    else: score -= min(10.0, (tiempo_espera - 120) / 60.0 * 2.0) # Penalización por espera larga
        
        duracion_fase_eval = info_estudio_paciente_eval["fases_duration"][fase]
        node_end_mins = node_mins + duracion_fase_eval
        
        medico_count_conflict = 0
        consulta_count_conflict = 0
        
        for v_node in self.visited:
            v_paciente, v_consulta, v_hora_str, v_medico, v_fase = v_node
            
            if v_paciente not in self.paciente_to_estudio_info: continue # Saltar si no hay info
            info_estudio_paciente_visitado = self.paciente_to_estudio_info[v_paciente]

            v_hora_parts = v_hora_str.split(':')
            v_mins = int(v_hora_parts[0]) * 60 + int(v_hora_parts[1])
            v_fase_duracion = info_estudio_paciente_visitado["fases_duration"][v_fase]
            v_end_mins = v_mins + v_fase_duracion
            
            # Comprobar superposición de tiempo entre node_to_evaluate y v_node
            # (startA < endB) and (startB < endA)
            if node_mins < v_end_mins and v_mins < node_end_mins:
                if v_medico == medico:
                    score -= 15.0
                    medico_count_conflict +=1
                if v_consulta == consulta:
                    score -= 15.0
                    consulta_count_conflict +=1
        
        # Si el recurso que se va a usar no genera conflictos, se da un pequeño bonus
        if medico_count_conflict == 0: score += 2.0
        if consulta_count_conflict == 0: score += 2.0
        
        return max(0.1, score) # Asegurar heurística positiva

    def move(self, node: Tuple):
        self.current_node = node
        self.visited.append(node)
        paciente, _, hora, _ , fase = node
        self.pacientes_progreso[paciente][fase] = hora # Guardar la fase y su hora
        
        # Verificar si la solución es completa y válida solo si todas las fases esperadas han sido asignadas
        num_total_fases_programadas = sum(len(fases) for fases in self.pacientes_progreso.values())
        
        num_total_fases_esperadas = 0
        for p_id in self.pacientes: # Iterar sobre la lista de todos los pacientes que deben ser programados
            if p_id in self.paciente_to_estudio_info:
                 num_total_fases_esperadas += len(self.paciente_to_estudio_info[p_id]["orden_fases"])
            # else: podría haber un paciente en self.pacientes que no esté en config? Improbable.

        if num_total_fases_programadas == num_total_fases_esperadas and num_total_fases_esperadas > 0 :
            self.valid_solution = all(
                self._fases_en_orden_correcto(p, prog_paciente)
                for p, prog_paciente in self.pacientes_progreso.items() if p in self.pacientes # Asegurar que solo se evalúan pacientes que debían ser programados
            )
        else:
            self.valid_solution = False


    def _fases_en_orden_correcto(self, paciente: str, fases_paciente_progreso: Dict) -> bool:
        """
        Verifica si las fases programadas para un paciente están completas y en el orden correcto
        según la definición de su estudio.
        fases_paciente_progreso es un dict {fase_nombre: hora_asignada}
        """
        if paciente not in self.paciente_to_estudio_info:
            return False # Paciente no tiene definición de estudio
        
        info_estudio = self.paciente_to_estudio_info[paciente]
        orden_fases_definicion = info_estudio["orden_fases"]
        
        # 1. Verificar si todas las fases definidas para el estudio del paciente han sido programadas
        if len(fases_paciente_progreso) != len(orden_fases_definicion):
            return False

        # 2. Verificar si las fases programadas son las correctas y están en orden
        # Crear una lista de tuplas (orden_definido, hora_programada) para las fases programadas
        fases_programadas_con_orden = []
        for fase_nombre, hora_asignada in fases_paciente_progreso.items():
            if fase_nombre not in orden_fases_definicion:
                return False # Fase programada no pertenece a la definición del estudio
            orden_definido = orden_fases_definicion[fase_nombre]
            fases_programadas_con_orden.append((orden_definido, hora_asignada, fase_nombre)) # Añadir fase_nombre para depuración
        
        # Ordenar por el orden definido en el estudio
        fases_programadas_con_orden.sort(key=lambda x: x[0])
        
        # Verificar secuencia de orden (1, 2, 3, ...)
        for i, (orden, _, _) in enumerate(fases_programadas_con_orden):
            if orden != i + 1:
                return False # El orden no es secuencial 1, 2, 3...
        
        
        return True
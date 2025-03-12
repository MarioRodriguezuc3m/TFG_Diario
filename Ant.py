import random
from generate_graph_components import *
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Graph import Graph  # Se usa solo para type hint

class Ant:
    def __init__(self, graph: "Graph", consultas_orden: Dict[str, int], consultas_duration: Dict[str, int],pacientes: List[str],alpha: float = 1.0, beta: float = 3.0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.visited: List[Tuple] = []
        self.consultas_orden= consultas_orden
        self.consultas_duration = consultas_duration
        self.pacientes = pacientes
        self.pacientes_progreso = defaultdict(dict,{paciente: {} for paciente in self.pacientes})
        self.current_node: Tuple = None
        self.total_cost: float = 0.0
        self.valid_solution = False

    def choose_next_node(self) -> Tuple:
      if self.current_node is None:
          # Elegir aleatoriamente el primer nodo (solo nodos iniciales válidos)
          valid_initial_nodes = [
              node for node in self.graph.nodes
              if self.consultas_orden[node[1]] == 1  # Solo ConsultaA (orden 1)
          ]
          return random.choice(valid_initial_nodes) if valid_initial_nodes else None
      else:
        current_paciente = self.current_node[0]
        current_consultas_completadas = self.pacientes_progreso[current_paciente].keys()
        current_consultas_requeridas = len(self.consultas_orden)
        # Verificar si el paciente actual ya completó todas sus consultas
        paciente_actual_completo = len(current_consultas_completadas) == current_consultas_requeridas
        
        candidates = self.graph.edges.get(self.current_node, [])
        
        if not candidates:
            return None
        
        candidate_list = []
        probabilities = []
        total = 0.0

        for node in candidates:
            # Iniciar con un acumulador de penalizaciones
            penalty = 0.0
            paciente, consulta, hora, _ = node

            # Penalizar cambiar de paciente si el actual no completó todas sus consultas
            if not paciente_actual_completo and paciente != current_paciente:
                penalty += 1.0

            # Penalizar si es un nuevo paciente que no inicia con la primera consulta
            if paciente != current_paciente and self.consultas_orden[consulta] != 1:
                penalty += 1.0

            # Para el mismo paciente, se requiere que la consulta sea la siguiente en orden
            if paciente == current_paciente:
                orden_actual = self.consultas_orden[self.current_node[1]]
                orden_siguiente = self.consultas_orden[consulta]
                if orden_siguiente != orden_actual + 1:
                    penalty += 1.0
                else:
                    # Además, la hora de la siguiente consulta debe ser posterior a la actual
                    hora_actual = datetime.datetime.strptime(self.current_node[2], "%H:%M")
                    hora_siguiente = datetime.datetime.strptime(hora, "%H:%M")
                    if hora_siguiente <= hora_actual:
                        penalty += 1.0

            # Penalizar si en el visited actual ya existe un nodo con el mismo médico a la misma hora
            for visited_node in self.visited:
                _, _, hora_v, medico_v = visited_node
                if medico_v == node[3] and hora_v == node[2]:
                    penalty += 1.0

            # Penalizar si en el visited actual ya existe un nodo con la misma consulta a la misma hora
            for visited_node in self.visited:
                _, consulta_v, hora_v, _ = visited_node
                if consulta_v == node[1] and hora_v == node[2]:
                    penalty += 1.0

            # Convertir la suma de penalizaciones en una heurística:
            # Un candidato sin penalizaciones tendrá heuristic = 1.0,
            # y a medida que se acumulen penalizaciones, la heurística disminuye (por ejemplo, 1/2, 1/3, etc.)
            heuristic = 1.0 / (1.0 + penalty)

            # Obtener el valor de la feromona para la transición actual
            pheromone = self.graph.get_pheromone(self.current_node, node)
            # Calcular el peso del candidato combinando feromonas y la heurística penalizada
            candidate_weight = (pheromone ** self.alpha) * (heuristic ** self.beta)
            candidate_list.append(node)
            probabilities.append(candidate_weight)
            total += candidate_weight

        # Si todos los candidatos reciben peso 0, se elige aleatoriamente entre ellos
        if total == 0:
            return random.choice(candidate_list)

        # Selección probabilística basada en los pesos normalizados
        return random.choices(candidate_list, weights=[p / total for p in probabilities], k=1)[0]

    def _consultas_en_orden_correcto(self, consultas_paciente: Dict) -> bool:
      """Verifica si las consultas de un paciente están en el orden correcto."""
      consultas = list(consultas_paciente.keys())
      # Obtener el orden de cada consulta según el grafo
      orden = [self.consultas_orden[c] for c in consultas]
      # Verificar que los órdenes sean secuenciales y ascendentes
      return orden == sorted(orden) and len(orden) == len(self.consultas_orden)

    def move(self, node: Tuple):
        self.current_node = node
        self.visited.append(node)
        paciente, consulta, hora, _ = node
        self.pacientes_progreso[paciente][consulta] = hora
        
        # Verificar solución completa y orden correcto
        self.valid_solution = all(
            self._consultas_en_orden_correcto(consultas) 
            for consultas in self.pacientes_progreso.values()
        )
    
    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        tiempos_pacientes = defaultdict(list)
        penalty = 0
        consultas_activas = []

        # Registro básico de tiempos
        for asignacion in asignaciones:
            paciente, consulta, hora_str, medico = asignacion
            hora_inicio = datetime.datetime.strptime(hora_str, "%H:%M").time()
            inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
            fin_min = inicio_min + self.consultas_duration[consulta]
            
            #Lista de con los nodos visitados para facilitar calculo de las restricciones
            consultas_activas.append({
                'paciente': paciente,
                'consulta': consulta,
                'inicio': inicio_min,
                'fin': fin_min,
                'medico': medico,
                'orden': self.consultas_orden[consulta]
            })
            tiempos_pacientes[paciente].append((self.consultas_orden[consulta], inicio_min, fin_min))

        # Chequeo de conflictos entre consultas
        for i in range(len(consultas_activas)):
            for j in range(i + 1, len(consultas_activas)):
                a = consultas_activas[i]
                b = consultas_activas[j]
                
                # Conflicto de médico
                if a['medico'] == b['medico'] and (a['inicio'] < b['fin'] and b['inicio'] < a['fin']):
                    # print(f"Conflicto de médico: {a} y {b}")
                    penalty += 5000
                    
                # Conflicto de tipo de consulta
                if a['consulta'] == b['consulta'] and (a['inicio'] < b['fin'] and b['inicio'] < a['fin']):
                    # print(f"Conflicto de consulta: {a} y {b}")
                    penalty += 5000

        # Verificación de orden y tiempo entre consultas del mismo paciente
        for paciente, tiempos in tiempos_pacientes.items():
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])  # x[0] = orden de consulta
            
            orden_esperado = 1
            for consulta_data in tiempos_ordenados:
                orden_actual = consulta_data[0]  # Obtener el orden de la tupla (orden, inicio, fin)
                if orden_actual != orden_esperado:
                    # print(f'Consulta fuera de orden: {paciente}, orden esperado {orden_esperado}, orden actual {orden_actual}')
                    penalty += 5000
                    break
                orden_esperado += 1
            
            # Verificar tiempo entre consultas (NUEVA LÓGICA)
            for i in range(1, len(tiempos_ordenados)):
                consulta_prev = tiempos_ordenados[i-1]
                consulta_actual = tiempos_ordenados[i]
                
                # Si la consulta actual empieza ANTES de que termine la anterior
                if consulta_actual[1] < consulta_prev[2]:
                    # print(f'tiempo entre consultas insuficiente: {paciente,consulta_prev,consulta_actual}')
                    penalty += 10000  # Penalización fuerte
                    break  # Dejamos de verificar este paciente
                else:
                    penalty += consulta_actual[1] - consulta_prev[2]  # Penalización por tiempo entre consultas

        return penalty
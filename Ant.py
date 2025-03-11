import random
from generate_graph_components import *
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Graph import Graph  # Se usa solo para type hint

class Ant:
    def __init__(self, graph: "Graph", alpha: float = 1.0, beta: float = 3.0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.visited: List[Tuple] = []
        self.pacientes_progreso = defaultdict(dict,{paciente: {} for paciente in self.graph.pacientes})
        self.current_node: Tuple = None
        self.total_cost: float = 0.0
        self.valid_solution = False

    def choose_next_node(self) -> Tuple:
      if self.current_node is None:
          # Elegir aleatoriamente el primer nodo (solo nodos iniciales válidos)
          valid_initial_nodes = [
              node for node in self.graph.nodes
              if self.graph.consultas_orden[node[1]] == 1  # Solo ConsultaA (orden 1)
          ]
          return random.choice(valid_initial_nodes) if valid_initial_nodes else None
      else:
        current_paciente = self.current_node[0]
        current_consultas_completadas = self.pacientes_progreso[current_paciente].keys()
        current_consultas_requeridas = len(self.graph.consultas_orden)
        
        # Verificar si el paciente actual ya completó todas sus consultas
        paciente_actual_completo = len(current_consultas_completadas) == current_consultas_requeridas
        
        candidates = self.graph.edges.get(self.current_node, [])
        
        valid_nodes = []
        for node in candidates:
            paciente, consulta, hora, _ = node
            
            # Permitir cambiar de paciente solo si el actual ya completó todo
            if not paciente_actual_completo and paciente != current_paciente:
                continue
                
            # Si es nuevo paciente, debe ser su primera consulta
            if paciente != current_paciente and self.graph.consultas_orden[consulta] != 1:
                continue
                
            # Restricciones para el mismo paciente
            if paciente == current_paciente:
                # Verificar orden y tiempo
                orden_actual = self.graph.consultas_orden[self.current_node[1]]
                orden_siguiente = self.graph.consultas_orden[consulta]
                
                if orden_siguiente != orden_actual + 1:
                    continue
                
                hora_actual = datetime.datetime.strptime(self.current_node[2], "%H:%M")
                hora_siguiente = datetime.datetime.strptime(hora, "%H:%M")
                
                if hora_siguiente <= hora_actual:
                    continue
            
            valid_nodes.append(node)
        
      if not valid_nodes:
          return None

      probabilities = []
      total = 0.0
        #current_time = self._get_current_time()
        
      for node in valid_nodes:
          pheromone = self.graph.get_pheromone(self.current_node, node)
          #tiempo_espera = self._calcular_espera_potencial(node, current_time)
          heuristic = 1 #/ (tiempo_espera + 1)  # +1 para evitar división por cero
          probabilities.append((pheromone ** self.alpha) * (heuristic ** self.beta))
          total += probabilities[-1]
      
      if total == 0:
          return random.choice(valid_nodes)
          
      return random.choices(valid_nodes, weights=[p/total for p in probabilities], k=1)[0]

    def _consultas_en_orden_correcto(self, consultas_paciente: Dict) -> bool:
      """Verifica si las consultas de un paciente están en el orden correcto."""
      consultas = list(consultas_paciente.keys())
      # Obtener el orden de cada consulta según el grafo
      orden = [self.graph.consultas_orden[c] for c in consultas]
      # Verificar que los órdenes sean secuenciales y ascendentes
      return orden == sorted(orden) and len(orden) == len(self.graph.consultas_orden)

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

    def _get_current_time(self):
        if not self.visited:
            return 0
        last_time_str = self.visited[-1][2]
        return datetime.datetime.strptime(last_time_str, "%H:%M").time()

    def _calcular_espera_potencial(self, node, current_time):
        _, _, nueva_hora_str, _ = node
        nueva_hora = datetime.datetime.strptime(nueva_hora_str, "%H:%M").time()
        diferencia = (nueva_hora.hour * 60 + nueva_hora.minute) - \
                    (current_time.hour * 60 + current_time.minute)
        return max(diferencia, 0)
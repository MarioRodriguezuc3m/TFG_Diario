import random
import matplotlib.pyplot as plt  # Asegúrate de tener matplotlib instalado
from Prueba_aco import *
from typing import List, Dict, Tuple, Set
from collections import defaultdict
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
        print("que es esto",self.pheromone.get(node1, {}).get(node2, 0.0))
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

class Ant:
    def __init__(self, graph: Graph, alpha: float = 1.0, beta: float = 3.0):
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
      print(consultas_paciente)
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

class ACO:
    def __init__(self, graph: Graph, n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
        self.graph = graph
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_solution = None
        self.best_cost = float('inf')

    def run(self):
        for _ in range(self.iterations):
            ants = [Ant(self.graph, self.alpha, self.beta) for _ in range(self.n_ants)]
            
            for ant in ants:
                while True:
                    next_node = ant.choose_next_node()
                    if next_node is None or ant.valid_solution:
                        break
                    ant.move(next_node)
                print(ant.valid_solution)
                if ant.valid_solution:
                    ant.total_cost = self.graph.calcular_tiempo_espera(ant.visited)
                    if ant.total_cost < self.best_cost:
                        self.best_cost = ant.total_cost
                        self.best_solution = ant.visited.copy()
                else:
                    ant.total_cost = float('inf')
            
            self.graph.update_pheromone(ants, self.rho, self.Q)
        
        return self.best_solution, self.best_cost

    def plot_convergence(self):
        plt.plot(self.best_distances)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.title('Convergencia del ACO')
        plt.show()

# Ejemplo de uso ---------------------------------------------------------------
if __name__ == "__main__":
  # Datos de entrada
  pacientes = ['Paciente1', 'Paciente2']
  consultas = ['ConsultaA', 'ConsultaB']
  horas = ['09:00', '10:00','11:00','12:00','13:00','14:00','15:00','16:00']
  medicos = ['MedicoX', 'MedicoY']
  orden_consultas = {'ConsultaA': 1, 'ConsultaB': 2}

  # Generar nodos y aristas
  nodos = generar_nodos(pacientes, consultas, horas, medicos)
  aristas = generar_aristas(nodos,orden_consultas)

  # Crear el grafo
  graph = Graph(nodos, aristas, orden_consultas, pacientes)

  # Configurar y ejecutar ACO
  aco = ACO(graph, n_ants=1, iterations=2, alpha=2.0, beta=1.0, rho=0.1, Q=1.0)
  best_solution, best_cost = aco.run()

  # Resultados
  print("Mejor solución encontrada:")
  for asignacion in best_solution:
      print(asignacion)
  print(f"Costo total: {best_cost}")
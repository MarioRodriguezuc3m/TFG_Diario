from Standard.ACO import ACO
from Standard.Graph import Graph
from utils.generate_graph_components import generar_nodos, generar_aristas
from utils.plot_gantt_solution import plot_gantt_chart
import json

def get_configuration():
   try:
      # Cargar la configuración desde el archivo JSON
      with open('/app/src/Standard/config.json', 'r') as file:
         config = json.load(file)
         pacientes = config['pacientes']
         consultas = config['consultas']
         horas = config['horas']
         medicos = config['medicos']
         fases = config['fases']
         orden_fases = config['orden_fases']
         fases_duration = config['fases_duration']
         return pacientes, consultas, horas, medicos, fases, orden_fases, fases_duration
   except Exception as e:
      print(f"Error al cargar la configuración: {e}")
      return [], [], [], [], [], {}, {}

# Ejemplo de uso ---------------------------------------------------------------
if __name__ == "__main__":
  # Se obtiene la configuración del problema desde el archivo config.json
  pacientes,consultas,horas,medicos,fases,orden_fases,fases_duration = get_configuration()
  # Generar nodos y aristas
  nodos = generar_nodos(pacientes, consultas, horas, medicos,fases)
  aristas = generar_aristas(nodos,orden_fases)

  # Crear el grafo
  graph = Graph(nodos, aristas, initial_pheromone=100.0)

  # Configurar y ejecutar ACO
  aco = ACO(graph, orden_fases, fases_duration, pacientes,medicos,consultas,horas, n_ants=10, iterations=100, alpha=1.0, beta=3.0, rho=0.5, Q=100000)
  best_solution, best_cost = aco.run()
  aco.plot_convergence()

# Resultados
  print("\n● Mejor solución encontrada:")
  for asignacion in best_solution:
      print(f"  - {asignacion}")
  print(f"\n● Costo total: {best_cost:.2f}")
  print(f"● Tiempo ejecución: {aco.execution_time:.2f}s")
  plot_gantt_chart(best_solution, fases_duration, pacientes, medicos, consultas)
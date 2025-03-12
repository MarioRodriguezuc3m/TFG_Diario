from ACO import ACO
from Graph import Graph
from generate_graph_components import generar_nodos, generar_aristas

# Ejemplo de uso ---------------------------------------------------------------
if __name__ == "__main__":
  # Datos de entrada
  pacientes = ['Paciente1', 'Paciente2']
  consultas = ['ConsultaA', 'ConsultaB']
  horas = ['09:00', '10:00','11:00','12:00','13:00','14:00','15:00','16:00']
  medicos = ['MedicoX', 'MedicoY']
  orden_consultas = {'ConsultaA': 1, 'ConsultaB': 2}
  consultas_duration = {'ConsultaA': 60, 'ConsultaB': 60}

  # Generar nodos y aristas
  nodos = generar_nodos(pacientes, consultas, horas, medicos)
  aristas = generar_aristas(nodos,orden_consultas)

  # Crear el grafo
  graph = Graph(nodos, aristas, pacientes)

  # Configurar y ejecutar ACO
  aco = ACO(graph, orden_consultas, consultas_duration, n_ants=5, iterations=10, alpha=2.0, beta=1.0, rho=0.1, Q=1.0)
  best_solution, best_cost = aco.run()

  # Resultados
  print("Mejor solución encontrada:")
  for asignacion in best_solution:
      print(asignacion)
  print(f"Costo total: {best_cost}")
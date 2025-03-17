from ACO_par import ACO_parallel
from Graph import Graph
from generate_graph_components import generar_nodos, generar_aristas

# Ejemplo de uso ---------------------------------------------------------------
if __name__ == "__main__":
  # Datos de entrada
  pacientes = ['Paciente1', 'Paciente2','Paciente3','Paciente4','Paciente5']
  consultas = ['ConsultaA', 'ConsultaB','ConsultaC','ConsultaD']
  horas = ['09:00', '10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00']
  medicos = ['MedicoX', 'MedicoY','MedicoZ']
  fases = ['Fase1', 'Fase2','Fase3','Fase4']
  orden_fases= {'Fase1': 1, 'Fase2': 2, 'Fase3': 3, 'Fase4': 4}
  fases_duration = {'Fase1': 60, 'Fase2': 60, 'Fase3': 60, 'Fase4': 60}

  # Generar nodos y aristas
  nodos = generar_nodos(pacientes, consultas, horas, medicos,fases)
  aristas = generar_aristas(nodos,orden_fases)

  # Crear el grafo
  graph = Graph(nodos, aristas)

  # Configurar y ejecutar ACO
  aco = ACO_parallel(graph, orden_fases, fases_duration, pacientes, n_ants=10, iterations=10, alpha=1.0, beta=3.0, rho=0.1, Q=1.0, num_parallel_ants=5)
  best_solution, best_cost = aco.run()
  aco.plot_convergence()

  # Resultados
  print("Mejor solución encontrada:")
  for asignacion in best_solution:
      print(asignacion)
  print(f"Costo total: {best_cost}")
  print(aco.get_execution_time())
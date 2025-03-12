from ACO import ACO
from Graph import Graph
from generate_graph_components import generar_nodos, generar_aristas

# Ejemplo de uso ---------------------------------------------------------------
if __name__ == "__main__":
  # Datos de entrada
  pacientes = ['Paciente1', 'Paciente2','Paciente3','Paciente4','Paciente5']
  consultas = ['ConsultaA', 'ConsultaB','ConsultaC','ConsultaD']
  horas = ['09:00', '10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00']
  medicos = ['MedicoX', 'MedicoY','MedicoZ']
  orden_consultas = {'ConsultaA': 1, 'ConsultaB': 2, 'ConsultaC': 3, 'ConsultaD': 4}
  consultas_duration = {'ConsultaA': 60, 'ConsultaB': 60, 'ConsultaC': 60, 'ConsultaD': 60}

  # Generar nodos y aristas
  nodos = generar_nodos(pacientes, consultas, horas, medicos)
  aristas = generar_aristas(nodos,orden_consultas)

  # Crear el grafo
  graph = Graph(nodos, aristas)

  # Configurar y ejecutar ACO
  aco = ACO(graph, orden_consultas, consultas_duration, pacientes, n_ants=15, iterations=300, alpha=1.0, beta=3.0, rho=0.1, Q=1.0)
  best_solution, best_cost = aco.run()
  aco.plot_convergence()

  # Resultados
  print("Mejor solución encontrada:")
  for asignacion in best_solution:
      print(asignacion)
  print(f"Costo total: {best_cost}")
import time
import matplotlib.pyplot as plt
from MinMax.MinMaxAco import MinMaxACO
from MinMax.MinMaxGraph import MinMaxGraph
from utils.generate_graph_components import generar_nodos, generar_aristas

def main():
    # Configuración del problema médico
    pacientes = ['Paciente1', 'Paciente2', 'Paciente3', 'Paciente4', 'Paciente5']
    consultas = ['ConsultaA', 'ConsultaB', 'ConsultaC']
    horas = ['09:00', '10:00', '11:00', '12:00', '13:00', 
            '14:00', '15:00', '16:00', '17:00', '18:00']
    medicos = ['MedicoX', 'MedicoY', 'MedicoZ']
    fases = ['Fase1', 'Fase2', 'Fase3', 'Fase4']
    
    orden_fases = {'Fase1': 1, 'Fase2': 2, 'Fase3': 3, 'Fase4': 4}
    duracion_fases = {'Fase1': 60, 'Fase2': 60, 'Fase3': 60, 'Fase4': 60}

    # Generar componentes del grafo
    nodos = generar_nodos(pacientes, consultas, horas, medicos, fases)
    aristas = generar_aristas(nodos, orden_fases)

    # Configurar grafo con límites estáticos
    graph = MinMaxGraph(
        nodes=nodos,
        edges=aristas,
        pheromone_max=65.0,
        pheromone_min=1.0,
        initial_pheromone=34.0
    )

    # Configurar algoritmo MinMaxACO
    aco = MinMaxACO(
        graph=graph,
        fases_orden=orden_fases,
        fases_duration=duracion_fases,
        pacientes=pacientes,
        medicos=medicos,
        consultas=consultas,
        horas=horas,
        n_ants=50,
        iterations=500,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        Q=286.0
    )

    # Ejecutar optimización
    mejor_solucion, mejor_costo = aco.run()

    # Mostrar resultados
    print("\n● Mejor solución encontrada:")
    for asignacion in mejor_solucion:
        print(f"  - {asignacion}")
    print(f"\n● Costo total: {mejor_costo:.2f}")
    print(f"● Tiempo ejecución: {aco.execution_time:.2f}s")

    aco.plot_convergence()

if __name__ == "__main__":
    main()
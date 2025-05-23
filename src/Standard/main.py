from Standard.ACO import ACO
from Standard.Graph import Graph 
from utils.generate_graph_components import generar_nodos, generar_aristas, construir_mapeo_paciente_info
from utils.plot_gantt_solution import plot_gantt_chart 

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import random

def get_configuration(config_path='/app/src/Standard/config.json'):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            # Verificar claves principales
            if "tipos_estudio" not in config or \
               "consultas" not in config or \
               "horas" not in config or \
               "medicos" not in config:
                print("Error: Faltan claves en la configuración.")
                return None
            # Verificar estructura de estudios
            for estudio in config["tipos_estudio"]:
                if not all(k in estudio for k in ["nombre_estudio", "pacientes", "fases", "orden_fases", "fases_duration"]):
                    print(f"Error: Estructura incorrecta en estudio {estudio.get('nombre_estudio', 'DESCONOCIDO')}")
                    return None
            return config
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON inválido en {config_path}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None

def generate_random_rgb_color_string():
    """Genera color RGB aleatorio para el gráfico"""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'rgb({r},{g},{b})'

if __name__ == "__main__":
    config_file_path = '/app/src/Standard/config.json'
    config_data = get_configuration(config_file_path)
    
    if config_data is None:
        print("No se pudo cargar la configuración.")
        exit(1)
        
    map_paciente_info = construir_mapeo_paciente_info(config_data['tipos_estudio'])
    
    # Generar componentes del grafo
    nodos = generar_nodos(config_data)
    if not nodos:
        print("Error generando nodos.")
        exit(1)
        
    aristas = generar_aristas(nodos, map_paciente_info)
    graph = Graph(nodos, aristas, initial_pheromone=1.0)
    
    # Configurar y ejecutar ACO
    aco = ACO(
        graph=graph,
        config_data=config_data,
        n_ants=20, iterations=50, alpha=1.0, beta=2.5, rho=0.2, Q=100.0
    )
    
    print("Ejecutando ACO...")
    best_solution, best_cost = aco.run()
    aco.plot_convergence()
    
    print("\n" + "="*50)
    print("RESULTADOS")
    print("="*50)
    
    if best_solution:
        print(f"\nMejor solución (Costo: {best_cost:.2f}):")
        
        # Agrupar asignaciones por paciente
        asignaciones_por_paciente = defaultdict(list)
        for asignacion_tuple in best_solution:
            paciente = asignacion_tuple[0]
            asignaciones_por_paciente[paciente].append(asignacion_tuple)
        
        # Mostrar resultados ordenados
        for paciente_id in sorted(asignaciones_por_paciente.keys()):
            asignaciones_paciente = asignaciones_por_paciente[paciente_id]
            print(f"\nPaciente: {paciente_id}")
            
            info_estudio_paciente = aco.paciente_to_estudio.get(paciente_id)
            if info_estudio_paciente:
                print(f"  Estudio: {info_estudio_paciente['nombre_estudio']}")
                
                # Ordenar fases según el orden del estudio
                asignaciones_ordenadas = sorted(
                    asignaciones_paciente, 
                    key=lambda asign_tuple: info_estudio_paciente['orden_fases'].get(asign_tuple[4], float('inf'))
                )
                
                for asign_tuple_ordenada in asignaciones_ordenadas:
                    _, consulta, hora, medico, fase = asign_tuple_ordenada
                    orden = info_estudio_paciente['orden_fases'].get(fase, "N/A")
                    duracion = info_estudio_paciente['fases_duration'].get(fase, "N/A")
                    print(f"  {orden}. {fase} - {hora} ({duracion}min) - {consulta} - {medico}")
            else:
                print(f"  Info de estudio no encontrada")
        
        print(f"\nCosto total: {best_cost:.2f}")
        if aco.execution_time is not None:
            print(f"Tiempo ejecución: {aco.execution_time:.2f}s")

        # Generar gráfico de Gantt
        if plot_gantt_chart:
            print("\nGenerando gráfico de Gantt...")
            try:
                # Preparar datos para el gráfico
                fases_duration_completo = {}
                all_configured_phase_names = set()
                
                for estudio_cfg in config_data['tipos_estudio']:
                    fases_duration_completo.update(estudio_cfg['fases_duration'])
                    all_configured_phase_names.update(estudio_cfg['fases'])

                # Colores aleatorios para cada fase
                dynamic_phase_colors_for_gantt = {}
                for phase_name in all_configured_phase_names:
                    dynamic_phase_colors_for_gantt[phase_name] = generate_random_rgb_color_string()
                
                # Lista completa de pacientes
                _pacientes_set = set()
                for estudio_cfg in config_data['tipos_estudio']:
                    _pacientes_set.update(estudio_cfg['pacientes'])
                lista_pacientes_completa = list(_pacientes_set)

                # Calcular horas de inicio y fin para el gráfico
                try:
                    min_h_str = min(config_data['horas'])
                    plot_start_hour = datetime.strptime(min_h_str, "%H:%M").hour
                    
                    max_h_str = max(config_data['horas'])
                    latest_slot_start_dt = datetime.strptime(max_h_str, "%H:%M")
                    
                    max_possible_duration = 0
                    if fases_duration_completo:
                        max_possible_duration = max(fases_duration_completo.values(), default=0)

                    potential_end_dt = datetime.combine(datetime.today(), latest_slot_start_dt.time()) + timedelta(minutes=max_possible_duration)
                    plot_end_hour = potential_end_dt.hour 
                    if potential_end_dt.minute > 0:
                        plot_end_hour += 1
                    if plot_end_hour > 23: 
                        plot_end_hour = 23

                except (ValueError, TypeError, IndexError):
                    print("Usando horas por defecto 8-20 para Gantt")
                    plot_start_hour = 8
                    plot_end_hour = 20

                gantt_output_dir = "/app/plots/"
                os.makedirs(gantt_output_dir, exist_ok=True)
                gantt_filepath = os.path.join(gantt_output_dir, "gantt_plotly_schedule.png")

                plot_gantt_chart(
                    best_solution=best_solution, 
                    fases_duration=fases_duration_completo,
                    pacientes=lista_pacientes_completa, 
                    medicos=config_data['medicos'], 
                    consultas=config_data['consultas'],
                    output_filepath=gantt_filepath,
                    phase_color_map=dynamic_phase_colors_for_gantt,
                    configured_start_hour=plot_start_hour,
                    configured_end_hour=plot_end_hour
                )
            except Exception as e:
                print(f"Error generando gráfico de Gantt: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\nNo se encontró solución válida.")
        if aco.execution_time is not None:
            print(f"Tiempo ejecución: {aco.execution_time:.2f}s")

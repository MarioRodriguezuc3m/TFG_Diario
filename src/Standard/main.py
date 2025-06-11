from Standard.ACO import ACO
from Standard.Graph import Graph 
from utils.generate_graph_components import generar_nodos, generar_aristas, construir_mapeo_paciente_info
from utils.plot_gantt_solution import plot_gantt_chart 

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, time
from typing import List

def get_configuration(config_path='/app/src/Standard/config.json'):
    """
    Carga y valida la configuración desde el archivo JSON especificado.
    Devuelve el diccionario de configuración si es válido, o None si hay errores.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            # Verificar claves
            expected_keys = ["tipos_estudio", "consultas", "hora_inicio", "hora_fin", 
                             "intervalo_consultas_minutos", "roles", "personal", "cargos"]
            if not all(key in config for key in expected_keys):
                print(f"Error: Faltan claves en la configuración. Esperadas: {', '.join(expected_keys)}.")
                return None

            # Validar tipo de intervalo_consultas_minutos
            if not isinstance(config["intervalo_consultas_minutos"], int) or config["intervalo_consultas_minutos"] <= 0:
                print("Error: 'intervalo_consultas_minutos' debe ser un entero positivo.")
                return None

            # Validar formato de horas
            try:
                datetime.strptime(config["hora_inicio"], "%H:%M")
                datetime.strptime(config["hora_fin"], "%H:%M")
            except ValueError:
                print("Error: 'hora_inicio' o 'hora_fin' tienen formato incorrecto. Usar HH:MM.")
                return None
            
            # Validar estructura de estudios
            for estudio in config["tipos_estudio"]:
                if not all(k in estudio for k in ["nombre_estudio", "pacientes", "fases", "orden_fases"]):
                    print(f"Error: Estructura incorrecta en estudio {estudio.get('nombre_estudio', 'DESCONOCIDO')}. Se esperan: nombre_estudio, pacientes, fases, orden_fases.")
                    return None

            # Validar roles, personal, cargos
            if not isinstance(config.get("roles"), list) or not all(isinstance(r, str) for r in config["roles"]):
                print("Error: 'roles' debe ser una lista de strings.")
                return None
            
            if not isinstance(config.get("personal"), dict):
                print("Error: 'personal' debe ser un diccionario.")
                return None
            for rol, cantidad in config["personal"].items():
                if rol not in config["roles"]:
                    print(f"Error: Rol '{rol}' en 'personal' no está definido en la lista 'roles'.")
                    return None
                if not isinstance(cantidad, int) or cantidad <= 0:
                    print(f"Error: La cantidad para el rol '{rol}' en 'personal' debe ser un entero positivo.")
                    return None

            if not isinstance(config.get("cargos"), dict):
                print("Error: 'cargos' debe ser un diccionario.")
                return None
            all_defined_phases_in_studies = set()
            for estudio_cfg in config["tipos_estudio"]:
                all_defined_phases_in_studies.update(estudio_cfg["fases"])

            for rol, fases_asignadas in config["cargos"].items():
                if rol not in config["roles"]:
                    print(f"Error: Rol '{rol}' en 'cargos' no está definido en la lista 'roles'.")
                    return None
                if not isinstance(fases_asignadas, list) or not all(isinstance(f, str) for f in fases_asignadas):
                    print(f"Error: Las fases para el rol '{rol}' en 'cargos' deben ser una lista de strings.")
                    return None
            
            phases_covered_by_roles = set()
            for rol in config["cargos"]:
                phases_covered_by_roles.update(config["cargos"][rol])
            
            for phase_study in all_defined_phases_in_studies:
                if phase_study not in phases_covered_by_roles:
                    print(f"Error crítico: La fase de estudio '{phase_study}' no puede ser realizada por ningún rol definido en 'cargos'.")
                    return None

            return config
    except Exception as e:
        print(f"Error inesperado al cargar configuración: {e}")
        import traceback
        traceback.print_exc()
        return None

def generar_horas_disponibles(hora_inicio_str: str, hora_fin_str: str, intervalo_minutos: int) -> List[str]:
    """
    Genera una lista de strings de tiempo ("HH:MM") entre hora_inicio y hora_fin con el intervalo dado.
    """
    horas = []
    try:
        start_time_obj = datetime.strptime(hora_inicio_str, "%H:%M").time()
        end_time_obj = datetime.strptime(hora_fin_str, "%H:%M").time() 
    except ValueError:
        print("Error: Formato de hora_inicio o hora_fin inválido en generar_horas_disponibles. Use HH:MM.")
        return []

    if intervalo_minutos <= 0:
        print("Error: intervalo_consultas_minutos debe ser positivo en generar_horas_disponibles.")
        return []

    current_dt = datetime.combine(datetime.today(), start_time_obj)
    end_datetime_limit = datetime.combine(datetime.today(), end_time_obj)

    while current_dt < end_datetime_limit:
        horas.append(current_dt.strftime("%H:%M"))
        current_dt += timedelta(minutes=intervalo_minutos)
    
    return horas

def get_aco_params(params_path='aco_params.json'):
    """
    Carga los parámetros del algoritmo ACO desde un archivo JSON y verifica que las claves sean correctas.
    """
    expected_keys = {"n_ants", "iterations", "alpha", "beta", "rho", "Q"}
    try:
        with open(params_path, 'r') as file:
            params = json.load(file)
        # Se verifican las claves
        params_keys = set(params.keys())
        if params_keys != expected_keys:
            raise Exception(f"Advertencia: Las claves del archivo de parámetros no son correctas.\n"
                  f"Esperadas: {sorted(expected_keys)}\n"
                  f"Encontradas: {sorted(params_keys)}\n")
        # Se validan los tipos de datos y valores
        if not isinstance(params["n_ants"], int) or params["n_ants"] <= 0:
            raise Exception("'n_ants' debe ser un entero positivo.")
        if not isinstance(params["iterations"], int) or params["iterations"] <= 0:
            raise Exception("'iterations' debe ser un entero positivo.")
        if not isinstance(params["alpha"], (int, float)) or params["alpha"] < 0:
            raise Exception("'alpha' debe ser un número no negativo.")
        if not isinstance(params["beta"], (int, float)) or params["beta"] < 0:
            raise Exception("'beta' debe ser un número no negativo.")
        if not isinstance(params["rho"], (int, float)) or not (0 < params["rho"] < 1):
            raise Exception("'rho' debe ser un número entre 0 y 1 (no inclusivo).")
        if not isinstance(params["Q"], (int, float)) or params["Q"] <= 0:
            raise Exception("'Q' debe ser un número positivo.")
        return params
    except Exception as e:
        raise Exception(f"Error cargando parámetros de ACO: {e}")

if __name__ == "__main__":
    config_file_path = '/app/src/Standard/config.json'
    aco_params_path = '/app/src/Standard/params_config.json'

    config_data = get_configuration(config_file_path)
    aco_params = get_aco_params(aco_params_path)

    if config_data is None:
        print("No se pudo cargar la configuración.")
        exit(1)
        
    # Definir nombre paciente 
    for i in range(len(config_data['tipos_estudio'])):
        estudio_config = config_data['tipos_estudio'][i]
        nombre_estudio = estudio_config.get("nombre_estudio", f"EstudioDesconocido_{i}")
        
        # Transformar nombres de pacientes
        if "pacientes" in estudio_config and isinstance(estudio_config["pacientes"], list):
            transformed_pacientes_list = []
            for p_generic in estudio_config["pacientes"]:
                transformed_name = f"{nombre_estudio}_{p_generic}"
                transformed_pacientes_list.append(transformed_name)
            
            config_data['tipos_estudio'][i]["pacientes"] = transformed_pacientes_list
        
    map_paciente_info = construir_mapeo_paciente_info(config_data['tipos_estudio'])
    
    # Generar horas disponibles
    horas_disponibles = generar_horas_disponibles(
        config_data['hora_inicio'],
        config_data['hora_fin'],
        config_data['intervalo_consultas_minutos']
    )

    if not horas_disponibles:
        print("Error: No se pudieron generar las horas disponibles. Revise la configuración de hora_inicio, hora_fin e intervalo.")
        exit(1)
    
    print(f"Horas disponibles generadas: {horas_disponibles}")

    lista_personal_instancias = []
    for rol, cantidad in config_data["personal"].items():
        for i in range(1, cantidad + 1):
            lista_personal_instancias.append(f"{rol}_{i}")
    print(f"Instancias de personal generadas: {lista_personal_instancias}")

    nodos = generar_nodos(config_data, horas_disponibles, lista_personal_instancias)
    if not nodos:
        print("Error generando nodos. Verifique que haya pacientes, fases, consultas, personal, horas disponibles y que los roles puedan realizar las fases.")
        exit(1)
        
    aristas = generar_aristas(nodos, map_paciente_info, duracion_consulta_minutos=config_data['intervalo_consultas_minutos'], horas_disponibles_str_list=horas_disponibles)
    graph = Graph(nodos, aristas, initial_pheromone=1.0)
    
    # Configurar y ejecutar ACO
    aco = ACO(
        graph=graph,
        config_data=config_data,
        horas_disponibles=horas_disponibles,
        lista_personal_instancias=lista_personal_instancias,
        n_ants=aco_params["n_ants"],
        iterations=aco_params["iterations"],
        alpha=aco_params["alpha"],
        beta=aco_params["beta"],
        rho=aco_params["rho"],
        Q=aco_params["Q"]
    )
    
    print("Ejecutando ACO...")
    best_solution, best_cost = aco.run()
    aco.plot_convergence()
    
    
    if best_solution:
        # Agrupar asignaciones por paciente
        asignaciones_por_paciente = defaultdict(list)
        for asignacion_tuple in best_solution:
            paciente = asignacion_tuple[0]
            asignaciones_por_paciente[paciente].append(asignacion_tuple)
        
        intervalo_global_min = config_data['intervalo_consultas_minutos']

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
                    _, consulta, hora, personal_asignado, fase = asign_tuple_ordenada
                    orden = info_estudio_paciente['orden_fases'].get(fase, "N/A")
                    duracion = intervalo_global_min
                    print(f"  {orden}. {fase} - {hora} ({duracion}min) - {consulta} - {personal_asignado}")
            else:
                print(f"  Información de estudio no encontrada para {paciente_id}")
        
        print(f"\nCosto total: {best_cost:.2f}")
        if aco.execution_time is not None:
            print(f"Tiempo de ejecución: {aco.execution_time:.2f}s")

        # Generar gráfico de Gantt
        if plot_gantt_chart:
            print("\nGenerando gráfico de Gantt...")
            try:
                # Preparar datos para el gráfico
                fases_duration_para_gantt = {}
                all_configured_phase_names = set()
                
                for estudio_cfg in config_data['tipos_estudio']:
                    all_configured_phase_names.update(estudio_cfg['fases'])

                # Asignar duración global a todas las fases
                for phase_name in all_configured_phase_names:
                    fases_duration_para_gantt[phase_name] = intervalo_global_min
                
                # Lista completa de pacientes
                _pacientes_set = set()
                for estudio_cfg in config_data['tipos_estudio']:
                    _pacientes_set.update(estudio_cfg['pacientes'])
                lista_pacientes_completa = list(_pacientes_set)

                # Calcular horas de inicio y fin para el gráfico
                try:
                    plot_start_hour = datetime.strptime(config_data['hora_inicio'], "%H:%M").hour
                    
                    latest_slot_start_str = horas_disponibles[-1]
                    latest_slot_start_dt_time = datetime.strptime(latest_slot_start_str, "%H:%M").time()
                    
                    plot_end_hour = datetime.combine(datetime.today(), latest_slot_start_dt_time) + timedelta(minutes=intervalo_global_min)

                    plot_end_hour_int = plot_end_hour.hour
                    if plot_end_hour.minute > 0:
                        plot_end_hour_int += 1
                    
                    if plot_end_hour_int > 23:
                        plot_end_hour_int = 23
                    
                    plot_end_hour = plot_end_hour_int

                except (ValueError, TypeError, IndexError) as e:
                    raise Exception(f"Error calculando horas para Gantt: {e}.")

                gantt_output_dir = "/app/plots/"
                os.makedirs(gantt_output_dir, exist_ok=True)
                gantt_filepath = os.path.join(gantt_output_dir, "gantt_plotly_schedule.png")

                plot_gantt_chart(
                    best_solution=best_solution, 
                    fases_duration=fases_duration_para_gantt,
                    pacientes=lista_pacientes_completa,
                    medicos=lista_personal_instancias,
                    consultas=config_data['consultas'],
                    output_filepath=gantt_filepath,
                    configured_start_hour=plot_start_hour,
                    configured_end_hour=plot_end_hour
                )
            except Exception as e:
                print(f"Error generando gráfico de Gantt: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\nNo se encontró una solución válida.")
        if aco.execution_time is not None:
            print(f"Tiempo de ejecución: {aco.execution_time:.2f}s")
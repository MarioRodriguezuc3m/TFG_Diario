import matplotlib.pyplot as plt 
# from Standard.Graph import Graph # Ya importado en Ant.py
from utils.Ant2 import Ant # Desde el mismo directorio de utils
from typing import Dict, List, Tuple
from collections import defaultdict
import datetime
import random
import time 
import os
import math

# Asegurarse de que Graph se pueda importar
try:
    from Standard.Graph import Graph
except ImportError:
    # Intentar una ruta relativa si está en un subdirectorio Standard dentro de src por ejemplo
    # Esto es solo un ejemplo, ajusta según tu estructura de proyecto real si Graph.py no está en PYTHONPATH
    # from ..Standard.Graph import Graph # Si ACO2.py está en src/utils y Graph.py en src/Standard
    pass


class ACO:
    def __init__(self, graph: Graph, config_data: Dict, n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
        self.graph = graph
        self.config_data = config_data
        
        self.tipos_estudio = config_data["tipos_estudio"]
        self.consultas = config_data["consultas"]
        self.horas = config_data["horas"]
        self.medicos = config_data["medicos"]
        
        # Mapeo de paciente a la información de su estudio (incluyendo orden y duración de fases)
        self.paciente_to_estudio = {} 
        self.pacientes = [] # Lista de todos los pacientes que deben ser programados
        
        for estudio in self.tipos_estudio:
            for paciente in estudio["pacientes"]:
                self.pacientes.append(paciente)
                self.paciente_to_estudio[paciente] = {
                    "nombre_estudio": estudio["nombre_estudio"],
                    "fases": estudio["fases"], # Lista de nombres de fases
                    "orden_fases": estudio["orden_fases"], # Dict {fase_nombre: orden_numerico}
                    "fases_duration": estudio["fases_duration"] # Dict {fase_nombre: duracion_minutos}
                }
        
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_solution = None
        self.total_costs = [] # Almacena el mejor costo de cada iteración
        self.best_cost = float('inf')
        self.execution_time = None

    def run(self):
        start_time = time.time()
        
        for iteration in range(self.iterations):
            ants = [Ant(self.graph, self.paciente_to_estudio, self.pacientes, 
                        self.alpha, self.beta) for _ in range(self.n_ants)]
            
            iteration_best_cost = float('inf')
            iteration_best_solution = None # La mejor solución de esta iteración
            # iteration_best_ant = None # No necesitamos guardar la hormiga, solo su solución y costo

            active_ants_solutions = [] # Almacenar (costo, solucion) para hormigas válidas

            for ant_idx, ant in enumerate(ants):
                # El número de pasos máximo debería ser suficiente para completar todas las fases de todos los pacientes
                # Por ejemplo: sum(len(info_estudio["fases"]) for info_estudio in self.paciente_to_estudio.values())
                # Más un pequeño margen. len(self.graph.nodes) es un límite muy suelto.
                max_steps = sum(len(self.paciente_to_estudio[p]["fases"]) for p in self.pacientes) * 2 # Un intento de límite razonable
                
                steps = 0
                while steps < max_steps and not ant.valid_solution:
                    next_node = ant.choose_next_node()
                    if next_node is None:
                        # La hormiga no puede encontrar un siguiente nodo válido
                        break 
                    ant.move(next_node)
                    steps += 1
                
                if ant.valid_solution:
                    cost = self.calcular_coste(ant.visited)
                    ant.total_cost = cost # Guardar costo en la hormiga para posible uso en update_pheromone
                    active_ants_solutions.append({'ant': ant, 'cost': cost, 'solution': ant.visited.copy()})
                    
                    if cost < iteration_best_cost:
                        iteration_best_cost = cost
                        iteration_best_solution = ant.visited.copy()
            
            # Actualizar feromonas basado en las soluciones de esta iteración
            # Usar todas las hormigas que encontraron una solución válida, o solo la mejor de la iteración
            # Aquí, actualizaremos con la mejor solución de la iteración si existe
            if iteration_best_solution is not None:
                # Aplicar búsqueda local a la mejor solución de la iteración
                current_solution_for_ls = iteration_best_solution
                current_cost_for_ls = iteration_best_cost
                
                # Búsqueda local (opcional, pero puede mejorar)
                # Hacer un número fijo de intentos de mejora o hasta que no mejore
                for _ in range(3): # Por ejemplo, 3 iteraciones de búsqueda local
                    improved_solution = self.local_search(current_solution_for_ls)
                    improved_cost = self.calcular_coste(improved_solution)
                    
                    # Aceptar mejora si es significativamente mejor (ej. 1%) o simplemente mejor
                    if improved_cost < current_cost_for_ls: # Considerar umbral como current_cost_for_ls * 0.99
                        current_solution_for_ls = improved_solution
                        current_cost_for_ls = improved_cost
                    else:
                        break # No más mejoras significativas
                
                # Si la búsqueda local mejoró la solución de la iteración
                if current_cost_for_ls < iteration_best_cost:
                    iteration_best_cost = current_cost_for_ls
                    iteration_best_solution = current_solution_for_ls
                
                # Actualizar la mejor solución global encontrada hasta ahora
                if iteration_best_cost < self.best_cost:
                    self.best_cost = iteration_best_cost
                    self.best_solution = iteration_best_solution.copy()

                # Actualizar feromonas usando la (posiblemente mejorada por LS) mejor solución de la iteración
                # Para el depósito de feromonas, podemos crear una "hormiga virtual" con esta solución
                # o pasar directamente la solución y su costo.
                # Graph.update_pheromone necesita una lista de hormigas.
                # Si solo actualizamos con la mejor de la iteración:
                if self.best_solution: # Asegurarse que hay una mejor solución global
                    # Crear una hormiga ficticia para la actualización de feromonas con la mejor solución de la iteración
                    temp_ant_for_pheromone = Ant(self.graph, self.paciente_to_estudio, self.pacientes, self.alpha, self.beta)
                    temp_ant_for_pheromone.visited = self.best_solution # Usar la mejor global para reforzar
                    temp_ant_for_pheromone.total_cost = self.best_cost # Usar la mejor global para reforzar
                    self.graph.update_pheromone([temp_ant_for_pheromone], self.rho, self.Q)

            else: # Ninguna hormiga encontró una solución válida en esta iteración
                # Podríamos evaporar feromonas de todas formas
                 self.graph.evaporate_pheromone(self.rho)


            self.total_costs.append(self.best_cost if self.best_cost != float('inf') else iteration_best_cost) # Registrar el mejor costo global o de iteración
            # print(f"Iteración {iteration + 1}/{self.iterations} - Mejor Costo Iter: {iteration_best_cost if iteration_best_cost != float('inf') else 'N/A'}, Mejor Costo Global: {self.best_cost if self.best_cost != float('inf') else 'N/A'}")

        end_time = time.time()
        self.execution_time = end_time - start_time
        
        return self.best_solution, self.best_cost

    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        if not asignaciones:
            return float('inf')

        fases_activas_detalle = [] # Lista de diccionarios con info detallada de cada asignación
        tiempos_pacientes = defaultdict(list) # {paciente: [(orden, inicio_min, fin_min, fase_nombre), ...]}
        hora_str_to_min_cache = {} # Cache para convertir "HH:MM" a minutos
        
        coste_total = 0.0

        # 1. Decodificar asignaciones y verificar validez básica
        for asignacion_idx, asignacion in enumerate(asignaciones):
            paciente, consulta, hora_str, medico, fase_nombre = asignacion
            
            if paciente not in self.paciente_to_estudio:
                coste_total += 50000 # Paciente no definido en la configuración
                continue 
            
            estudio_info = self.paciente_to_estudio[paciente]
            
            if fase_nombre not in estudio_info["fases"]:
                coste_total += 40000 # Fase no pertenece al estudio del paciente
                continue

            if hora_str not in hora_str_to_min_cache:
                try:
                    hora_obj = datetime.datetime.strptime(hora_str, "%H:%M").time()
                    inicio_min = hora_obj.hour * 60 + hora_obj.minute
                    hora_str_to_min_cache[hora_str] = inicio_min
                except ValueError:
                    coste_total += 60000 # Formato de hora inválido
                    continue
            else:
                inicio_min = hora_str_to_min_cache[hora_str]
                
            duracion = estudio_info["fases_duration"].get(fase_nombre)
            if duracion is None:
                coste_total += 45000 # Duración de fase no definida
                continue
            
            fin_min = inicio_min + duracion
            orden_fase = estudio_info["orden_fases"].get(fase_nombre)
            if orden_fase is None:
                coste_total += 46000 # Orden de fase no definido
                continue

            fases_activas_detalle.append({
                'paciente': paciente, 'consulta': consulta, 'medico': medico, 'fase': fase_nombre,
                'inicio_min': inicio_min, 'fin_min': fin_min, 'orden': orden_fase,
                'original_tuple': asignacion, 'idx_original': asignacion_idx
            })
            tiempos_pacientes[paciente].append((orden_fase, inicio_min, fin_min, fase_nombre))

        if coste_total > 0: # Si ya hay errores graves de definición, retornar temprano
            return coste_total

        # 2. Penalizaciones por conflictos de recursos
        # Usar un enfoque de barrido temporal para detectar solapamientos
        # Crear lista de eventos (inicio_fase, fin_fase)
        eventos = []
        for i, f_activa in enumerate(fases_activas_detalle):
            eventos.append((f_activa['inicio_min'], 'start', i, f_activa['medico'], f_activa['consulta']))
            eventos.append((f_activa['fin_min'], 'end', i, f_activa['medico'], f_activa['consulta']))
        
        eventos.sort()

        medicos_ocupados = defaultdict(int) # medico -> contador de fases activas
        consultas_ocupadas = defaultdict(int) # consulta -> contador de fases activas

        for t, tipo_evento, idx_fase, medico_evento, consulta_evento in eventos:
            if tipo_evento == 'start':
                if medicos_ocupados[medico_evento] > 0:
                    coste_total += 2000 # Conflicto de médico
                medicos_ocupados[medico_evento] += 1
                
                if consultas_ocupadas[consulta_evento] > 0:
                    coste_total += 2000 # Conflicto de consulta
                consultas_ocupadas[consulta_evento] += 1
            else: # 'end'
                medicos_ocupados[medico_evento] -=1
                consultas_ocupadas[consulta_evento] -=1

        # 3. Penalizaciones por secuencia y tiempos de espera por paciente
        for paciente, fases_programadas_paciente in tiempos_pacientes.items():
            estudio_info = self.paciente_to_estudio[paciente]
            fases_programadas_paciente.sort(key=lambda x: x[0]) # Ordenar por orden de fase definido

            # Verificar si todas las fases del estudio están presentes
            num_fases_definidas = len(estudio_info["orden_fases"])
            if len(fases_programadas_paciente) != num_fases_definidas:
                coste_total += 15000 * abs(num_fases_definidas - len(fases_programadas_paciente)) # Penalización por fases faltantes/extras
            
            orden_esperado = 1
            fin_fase_anterior_min = -1

            for orden_actual, inicio_actual_min, fin_actual_min, fase_nombre_actual in fases_programadas_paciente:
                # Verificar orden secuencial correcto
                if orden_actual != orden_esperado:
                    coste_total += 10000 # Penalización por orden incorrecto
                
                if fin_fase_anterior_min != -1: # Si no es la primera fase del paciente
                    if inicio_actual_min < fin_fase_anterior_min:
                        # Solapamiento entre fases del mismo paciente
                        coste_total += 5000 * (fin_fase_anterior_min - inicio_actual_min)
                    else:
                        # Penalización por tiempo de espera entre fases
                        tiempo_espera = inicio_actual_min - fin_fase_anterior_min
                        if tiempo_espera > 120: # Más de 2 horas de espera
                            coste_total += (tiempo_espera - 120) * 2 # Penalización creciente
                        elif tiempo_espera > 15: # Pequeña penalización por cualquier espera > 15 min
                             coste_total += tiempo_espera * 0.5
                
                fin_fase_anterior_min = fin_actual_min
                orden_esperado += 1
        
        # 4. (Opcional) Costo por duración total del plan (makespan) o uso de recursos
        # Por ejemplo, el tiempo hasta que la última fase de cualquier paciente termina.
        if fases_activas_detalle:
            max_fin_tiempo = max(f['fin_min'] for f in fases_activas_detalle if 'fin_min' in f)
            # coste_total += max_fin_tiempo * 0.01 # Pequeña penalización para favorecer soluciones más cortas
        
        return coste_total if coste_total > 0 else 0.1 # Evitar coste cero si todo es perfecto


    def local_search(self, solution: List[Tuple]) -> List[Tuple]:
        # Esta es una implementación simplificada. Una búsqueda local más robusta
        # podría probar diferentes vecindarios (cambiar hora, médico, consulta, orden de dos tareas).
        current_best_solution = solution
        current_best_cost = self.calcular_coste(solution)

        if current_best_cost == 0.1: # Ya es óptima según la función de coste
            return solution

        # Intentar algunas modificaciones aleatorias simples
        # Por ejemplo, tomar una asignación y cambiar su hora, médico o consulta
        # O intercambiar dos asignaciones
        for _ in range(min(len(solution) * 2, 20)): # Intentar un número limitado de movimientos
            if not current_best_solution: break
            
            temp_solution = list(current_best_solution) # Copiar
            idx_to_change = random.randrange(len(temp_solution))
            
            paciente, consulta, hora_str, medico, fase = temp_solution[idx_to_change]

            # Estrategia: Cambiar un elemento aleatorio
            change_type = random.choice(["hora", "medico", "consulta"])
            
            new_asig = None
            if change_type == "hora":
                nueva_hora = random.choice(self.horas)
                if nueva_hora != hora_str:
                    new_asig = (paciente, consulta, nueva_hora, medico, fase)
            elif change_type == "medico":
                nuevo_medico = random.choice(self.medicos)
                if nuevo_medico != medico:
                    new_asig = (paciente, consulta, hora_str, nuevo_medico, fase)
            elif change_type == "consulta":
                nueva_consulta = random.choice(self.consultas)
                if nueva_consulta != consulta:
                    new_asig = (paciente, nueva_consulta, hora_str, medico, fase)
            
            if new_asig:
                original_asig = temp_solution[idx_to_change]
                temp_solution[idx_to_change] = new_asig
                new_cost = self.calcular_coste(temp_solution)
                if new_cost < current_best_cost:
                    current_best_cost = new_cost
                    current_best_solution = temp_solution # Aceptar la mejora
                # No revertimos si no mejora, esto es más como un "random walk" con aceptación de mejoras
        
        # Podrías añadir aquí las funciones fix_conflicts si tienes una forma robusta de identificar
        # conflictos específicos y aplicar reparaciones dirigidas.
        # El código de find_all_conflicts y fix_conflicts necesitaría una revisión cuidadosa
        # para integrarse bien con la lógica de múltiples estudios y la función de coste.
        # Por simplicidad, el ejemplo anterior de local_search con fix_conflicts no se incluye aquí
        # para mantener el foco en los cambios principales de multi-estudio.

        return current_best_solution


    # Las funciones find_all_conflicts, fix_conflicts, _get_previous_phase_end, _try_random_improvements
    # necesitarían una adaptación cuidadosa si se reintroducen, asegurando que siempre
    # usan self.paciente_to_estudio para obtener información específica del estudio.
    # Por ejemplo, _get_previous_phase_end:
    def _get_previous_phase_end(self, solution_details: List[Dict], paciente: str, current_fase_orden: int) -> int:
        """
        Encuentra el tiempo de finalización de la fase anterior del mismo paciente.
        solution_details es una lista de dicts como la generada en calcular_coste (fases_activas_detalle).
        current_fase_orden es el orden numérico de la fase actual.
        Devuelve la hora de finalización en minutos, o 0 si es la primera fase.
        """
        if current_fase_orden == 1:
            return 0 # Primera fase, no hay anterior

        prev_orden = current_fase_orden - 1
        for detalle_asig in solution_details:
            if detalle_asig['paciente'] == paciente and detalle_asig['orden'] == prev_orden:
                return detalle_asig['fin_min']
        return 0 # Fase anterior no encontrada (no debería pasar en una solución válida parcial)

    def plot_convergence(self):
        if not self.total_costs:
            print("No hay datos de costos para graficar la convergencia.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.total_costs, marker='o', linestyle='-')
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Costo Encontrado')
        plt.title('Convergencia del Algoritmo ACO')
        plt.grid(True)
        
        # Asegurar que el directorio de plots existe
        plot_dir = "/app/plots" # Asumiendo que corre en un contenedor Docker con esta ruta
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            plt.savefig(os.path.join(plot_dir, "convergencia_aco.png"))
            print(f"Gráfico de convergencia guardado en {os.path.join(plot_dir, 'convergencia_aco.png')}")
        except Exception as e:
            print(f"Error al guardar el gráfico de convergencia: {e}")
        plt.close()

    def get_execution_time(self):
        return self.execution_time
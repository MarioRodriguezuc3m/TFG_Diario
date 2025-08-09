import matplotlib.pyplot as plt 
from utils.Ant import Ant
from typing import Dict, List, Tuple
from collections import defaultdict
import datetime
import random
import time 
import os
import math
try:
    from Standard.Graph import Graph
except ImportError:
    pass


class ACO:
    def __init__(self, graph: Graph, config_data: Dict, horas_disponibles: List, 
                 lista_personal_instancias: List[str], # Nueva entrada
                 n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
        """
        Inicializa el objeto ACO con los parámetros y datos necesarios.
        """
        self.graph = graph
        self.config_data = config_data
        
        self.tipos_estudio = config_data["tipos_estudio"]
        self.consultas = config_data["consultas"]
        self.horas = horas_disponibles
        self.duracion_consultas = config_data.get("intervalo_consultas_minutos")
        
        # Atributos relacionados con roles y personal
        self.roles_config = config_data["roles"]
        self.personal_cantidad_config = config_data["personal"]
        self.cargos_config = config_data["cargos"]
        self.lista_personal_instancias = lista_personal_instancias

        # Mapeo de fase a roles que pueden realizarla
        self.fase_a_roles_compatibles = defaultdict(list)
        for rol, fases_asignadas in self.cargos_config.items():
            for fase in fases_asignadas:
                self.fase_a_roles_compatibles[fase].append(rol)
        
        self.paciente_to_estudio = {} 
        _unique_pacientes_set = set()

        for estudio in self.tipos_estudio:
            for paciente in estudio["pacientes"]:
                _unique_pacientes_set.add(paciente)
                self.paciente_to_estudio[paciente] = {
                    "nombre_estudio": estudio["nombre_estudio"],
                    "fases": estudio["fases"], 
                    "orden_fases": estudio["orden_fases"],
                }
        self.pacientes = list(_unique_pacientes_set)
        
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_solution = None
        self.total_costs = [] 
        self.best_cost = float('inf')
        self.execution_time = None

    def run(self):
        """
        Ejecuta el ciclo principal del algoritmo ACO, incluyendo la búsqueda local y la actualización de feromonas.
        """
        tiempo_inicio = time.time()
        
        for iteration in range(self.iterations):
            ants = [Ant(self.graph, self.paciente_to_estudio, self.pacientes,self.duracion_consultas,
                        self.alpha, self.beta) for _ in range(self.n_ants)]
            
            iteration_best_cost = float('inf')
            iteration_best_solution = None

            active_ants_solutions = []

            for ant_idx, ant in enumerate(ants):
                # Límite de pasos para evitar bucles infinitos
                max_steps = sum(len(self.paciente_to_estudio[p]["fases"]) for p in self.pacientes) * 2
                
                steps = 0
                while steps < max_steps and not ant.valid_solution:
                    next_node = ant.choose_next_node()
                    if next_node is None:
                        break 
                    ant.move(next_node)
                    steps += 1
                
                if ant.valid_solution:
                    cost = self.calcular_coste(ant.visited)
                    ant.total_cost = cost
                    active_ants_solutions.append({'ant': ant, 'cost': cost, 'solution': ant.visited.copy()})
                    
                    if cost < iteration_best_cost:
                        iteration_best_cost = cost
                        iteration_best_solution = ant.visited.copy()
            
            # Aplicar búsqueda local a la mejor solución de la iteración
            if iteration_best_solution is not None:
                current_solution_for_ls = iteration_best_solution
                current_cost_for_ls = iteration_best_cost
                
                # Intentar mejoras locales
                for _ in range(3):
                    improved_solution = self.local_search(current_solution_for_ls)
                    improved_cost = self.calcular_coste(improved_solution)
                    
                    if improved_cost < current_cost_for_ls:
                        current_solution_for_ls = improved_solution
                        current_cost_for_ls = improved_cost
                    else:
                        break
                
                # Actualizar si la búsqueda local mejoró
                if current_cost_for_ls < iteration_best_cost:
                    iteration_best_cost = current_cost_for_ls
                    iteration_best_solution = current_solution_for_ls
                
                # Actualizar mejor solución global
                if iteration_best_cost < self.best_cost:
                    self.best_cost = iteration_best_cost
                    self.best_solution = iteration_best_solution.copy()

                if self.best_solution:
                    temp_ant_for_pheromone = Ant(self.graph, self.paciente_to_estudio, self.pacientes, 
                                                 self.duracion_consultas, self.alpha, self.beta)
                    temp_ant_for_pheromone.visited = self.best_solution
                    temp_ant_for_pheromone.total_cost = self.best_cost
                    temp_ant_for_pheromone.valid_solution = True
                    self.graph.update_pheromone([temp_ant_for_pheromone], self.rho, self.Q)

            else:
                # Evaporar feromonas si no hay solución válida
                self.graph.update_pheromone([], self.rho, self.Q)

            if iteration % 10 == 0:
                print(f"Iteración {iteration}/{self.iterations} - Mejor: {self.best_cost:.2f}")
            self.total_costs.append(self.best_cost if self.best_cost != float('inf') else iteration_best_cost)

        tiempo_fin = time.time()
        self.execution_time = tiempo_fin - tiempo_inicio
        
        return self.best_solution, self.best_cost

    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        """
        Calcula el coste total de una solución evaluando múltiples criterios:
        1. Validez de asignaciones (pacientes, fases, horas, duraciones)
        2. Conflictos de recursos (médicos y consultas ocupados simultáneamente)
        3. Secuencia correcta de fases por paciente
        4. Tiempos de espera entre fases del mismo paciente
        """
        if not asignaciones:
            return float('inf')

        fases_activas_detalle = []
        tiempos_pacientes = defaultdict(list) 
        hora_str_to_min_cache = {}
        
        coste_total = 0.0

        # PASO 1: Validar cada asignación individual
        for asignacion_idx, asignacion in enumerate(asignaciones):
            paciente, consulta, hora_str, personal_instancia, fase_nombre = asignacion
            
            if paciente not in self.paciente_to_estudio:
                coste_total += 50000  # Penalización grave por paciente inexistente
                continue 
            
            estudio_info = self.paciente_to_estudio[paciente]
            
            if fase_nombre not in estudio_info["fases"]:
                coste_total += 40000  # Penalización por fase incorrecta
                continue

            if hora_str not in hora_str_to_min_cache:
                try:
                    hora_obj = datetime.datetime.strptime(hora_str, "%H:%M").time()
                    inicio_min = hora_obj.hour * 60 + hora_obj.minute
                    hora_str_to_min_cache[hora_str] = inicio_min
                except ValueError:
                    coste_total += 60000  # Penalización por formato de hora inválido
                    continue
            else:
                inicio_min = hora_str_to_min_cache[hora_str]
                
            fin_min = inicio_min + self.duracion_consultas
            orden_fase = estudio_info["orden_fases"].get(fase_nombre)
            if orden_fase is None:
                coste_total += 46000 
                continue
            
            # Validar que el personal asignado puede realizar la fase
            rol_asignado = personal_instancia.split('_')[0]
            if rol_asignado not in self.cargos_config or fase_nombre not in self.cargos_config[rol_asignado]:
                coste_total += 70000 # Penalización por personal incorrecto para la fase
                continue


            fases_activas_detalle.append({
                'paciente': paciente, 'consulta': consulta, 'personal': personal_instancia, 'fase': fase_nombre,
                'inicio_min': inicio_min, 'fin_min': fin_min, 'orden': orden_fase,
                'original_tuple': asignacion, 'idx_original': asignacion_idx
            })
            tiempos_pacientes[paciente].append((orden_fase, inicio_min, fin_min, fase_nombre))

        if coste_total > 0: # Si hay errores graves de validación temprana
            return coste_total

        # PASO 2: Detectar conflictos de recursos mediante barrido
        eventos = []
        # Crear eventos de inicio y fin para cada fase
        for i, f_activa in enumerate(fases_activas_detalle):
            eventos.append((f_activa['inicio_min'], 'start', i, f_activa['personal'], f_activa['consulta']))
            eventos.append((f_activa['fin_min'], 'end', i, f_activa['personal'], f_activa['consulta']))
        
        eventos.sort()

        personal_ocupado = defaultdict(int)
        consultas_ocupadas = defaultdict(int)

        for t, tipo_evento, idx_fase, personal_evento, consulta_evento in eventos:
            if tipo_evento == 'start':
                if personal_ocupado[personal_evento] > 0:
                    coste_total += 100000  # Personal ya ocupado
                personal_ocupado[personal_evento] += 1
                
                if consultas_ocupadas[consulta_evento] > 0:
                    coste_total += 100000  # Consulta ya ocupada
                consultas_ocupadas[consulta_evento] += 1
            else:
                personal_ocupado[personal_evento] -= 1
                consultas_ocupadas[consulta_evento] -= 1

        # PASO 3: Verificar secuencia y tiempos por paciente
        for paciente, fases_programadas_paciente in tiempos_pacientes.items():
            estudio_info = self.paciente_to_estudio[paciente]
            fases_programadas_paciente.sort(key=lambda x: x[0])

            # Verificar que están todas las fases del estudio
            num_fases_definidas = len(estudio_info["orden_fases"])
            if len(fases_programadas_paciente) != num_fases_definidas:
                coste_total += 15000 * abs(num_fases_definidas - len(fases_programadas_paciente))
            
            orden_esperado = 1
            fin_fase_anterior_min = -1

            for orden_actual, inicio_actual_min, fin_actual_min, fase_nombre_actual in fases_programadas_paciente:
                # Verificar que las fases están en el orden correcto
                if orden_actual != orden_esperado:
                    coste_total += 100000  # Penalización por orden incorrecto
                
                if fin_fase_anterior_min != -1:  # No es la primera fase
                    if inicio_actual_min < fin_fase_anterior_min:
                        # Las fases se solapan
                        coste_total += 5000 * (fin_fase_anterior_min - inicio_actual_min)
                    else:
                        # Calcular tiempo de espera entre fases
                        tiempo_espera = inicio_actual_min - fin_fase_anterior_min
                        if tiempo_espera > 120:  # Más de 2 horas de espera
                            coste_total += (tiempo_espera - 120) * 2  # Penalización creciente
                        elif tiempo_espera > 15:  # Más de 15 minutos
                             coste_total += tiempo_espera * 0.5  # Penalización leve
                
                fin_fase_anterior_min = fin_actual_min
                orden_esperado += 1
        
        return coste_total if coste_total > 0 else 0.1  # Evitar coste cero
        
    def _identificar_asignaciones_conflictivas(self, solution: List[Tuple]) -> List[int]:
        """
        Identifica los índices de las asignaciones en la solución que tienen
        conflictos directos de recursos (personal o consulta ocupados).
        """
        conflictive_indices = set()
        if not solution or len(solution) < 2:
            return []

        # Cache para conversiones de hora y duraciones para eficiencia
        hora_str_to_min_cache = {}
        duracion_consulta_min = self.duracion_consultas

        # Convertir todas las asignaciones a un formato más manejable
        processed_assignments = []
        for i, asignacion in enumerate(solution):
            paciente, consulta, hora_str, personal_instancia, fase = asignacion

            if hora_str not in hora_str_to_min_cache:
                try:
                    hora_obj = datetime.datetime.strptime(hora_str, "%H:%M").time()
                    inicio_min = hora_obj.hour * 60 + hora_obj.minute
                    hora_str_to_min_cache[hora_str] = inicio_min
                except ValueError:
                    continue 
            else:
                inicio_min = hora_str_to_min_cache[hora_str]

            fin_min = inicio_min + duracion_consulta_min
            processed_assignments.append({
                'idx': i, 'paciente': paciente, 'consulta': consulta,
                'personal': personal_instancia, 'fase': fase,
                'inicio_min': inicio_min, 'fin_min': fin_min,
                'original_tuple': asignacion
            })

        # Comprobar conflictos entre pares de asignaciones
        for i in range(len(processed_assignments)):
            asig1 = processed_assignments[i]
            for j in range(i + 1, len(processed_assignments)):
                asig2 = processed_assignments[j]

                # Comprobar solapamiento temporal
                overlap = (asig1['inicio_min'] < asig2['fin_min'] and
                        asig2['inicio_min'] < asig1['fin_min'])

                if overlap:
                    # Conflicto de personal (diferentes pacientes, misma instancia de personal)
                    if asig1['personal'] == asig2['personal'] and asig1['paciente'] != asig2['paciente']:
                        conflictive_indices.add(asig1['idx'])
                        conflictive_indices.add(asig2['idx'])

                    # Conflicto de consulta (diferentes pacientes, misma consulta)
                    if asig1['consulta'] == asig2['consulta'] and asig1['paciente'] != asig2['paciente']:
                        conflictive_indices.add(asig1['idx'])
                        conflictive_indices.add(asig2['idx'])

        return list(conflictive_indices)

    def local_search(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Realiza una búsqueda local sobre la solución para intentar mejorar el coste.
        """
        current_best_solution = list(solution) 
        current_best_cost = self.calcular_coste(current_best_solution)

        if not current_best_solution or current_best_cost == 0.1: # Coste mínimo sin penalizaciones
            return current_best_solution

        num_improvement_attempts = 15 

        for attempt in range(num_improvement_attempts):
            if not current_best_solution: break
            
            temp_solution = list(current_best_solution) 
            conflictive_indices = self._identificar_asignaciones_conflictivas(temp_solution)
            
            idx_to_change = -1
            if conflictive_indices and random.random() < 0.9: 
                idx_to_change = random.choice(conflictive_indices)
            else: 
                if not temp_solution: continue
                idx_to_change = random.randrange(len(temp_solution))

            if idx_to_change == -1 or idx_to_change >= len(temp_solution): continue

            original_assignment = temp_solution[idx_to_change]
            # personal_actual_instancia es el 4to elemento
            paciente, consulta, hora_str, personal_actual_instancia, fase = original_assignment

            # 2. Intentar cambiar un elemento aleatorio de la asignación seleccionada
            change_options = []
            # Cambiar hora
            available_new_horas = [h for h in self.horas if h != hora_str]
            if available_new_horas:
                change_options.append(("hora", random.choice(available_new_horas)))
            
            
            roles_compatibles_con_fase = self.fase_a_roles_compatibles.get(fase, [])
            
            available_new_personal_instancias = []
            if roles_compatibles_con_fase:
                for p_inst in self.lista_personal_instancias:
                    p_rol = p_inst.split('_')[0]
                    if p_rol in roles_compatibles_con_fase and p_inst != personal_actual_instancia:
                        available_new_personal_instancias.append(p_inst)
            
            # Cambiar personal_instancia
            if available_new_personal_instancias:
                change_options.append(("personal", random.choice(available_new_personal_instancias)))

            available_new_consultas = [c for c in self.consultas if c != consulta]
            
            # Cambiar consulta
            if available_new_consultas:
                change_options.append(("consulta", random.choice(available_new_consultas)))

            if not change_options: continue

            change_type, new_value = random.choice(change_options)
            
            new_asig = None
            if change_type == "hora":
                new_asig = (paciente, consulta, new_value, personal_actual_instancia, fase)
            elif change_type == "personal": 
                new_asig = (paciente, consulta, hora_str, new_value, fase)
            elif change_type == "consulta":
                new_asig = (paciente, new_value, hora_str, personal_actual_instancia, fase)
            
            if new_asig:
                modified_solution_attempt = list(temp_solution)
                modified_solution_attempt[idx_to_change] = new_asig
                new_cost = self.calcular_coste(modified_solution_attempt)
                
                if new_cost < current_best_cost:
                    current_best_cost = new_cost
                    current_best_solution = modified_solution_attempt

        return current_best_solution

    def plot_convergence(self, output_dir: str = "/app/plots"):
        """
        Genera y guarda un gráfico de la convergencia del algoritmo ACO.
        """
        if not self.total_costs:
            print("No hay datos para graficar convergencia.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.total_costs, marker='o', linestyle='-')
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Costo Encontrado')
        plt.title('Convergencia del Algoritmo ACO')
        plt.grid(True)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            plt.savefig(os.path.join(output_dir, "convergencia_aco.png"))
            print(f"Gráfico guardado en {os.path.join(output_dir, 'convergencia_aco.png')}")
        except Exception as e:
            print(f"Error guardando gráfico: {e}")
        plt.close()

    def get_execution_time(self):
        """
        Devuelve el tiempo de ejecución del algoritmo.
        """
        return self.execution_time
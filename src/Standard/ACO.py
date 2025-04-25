import matplotlib.pyplot as plt 
from Standard.Graph import Graph
from utils.Ant import Ant
from typing import Dict, List, Tuple
from collections import defaultdict
import datetime
import random
import time 
import os
import math

class ACO:
    def __init__(self, graph: Graph, fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str], medicos: List[str], consultas: List[str], horas: List[str], n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
        """
        Inicializa el algoritmo de Optimización por Colonia de Hormigas.
        
        Parámetros:
        - graph: Grafo que representa el espacio de búsqueda
        - fases_orden: Diccionario con el orden de las fases
        - fases_duration: Diccionario con la duración de cada fase en minutos
        - pacientes: Lista de pacientes a programar
        - medicos: Lista de médicos disponibles
        - consultas: Lista de salas de consulta disponibles
        - horas: Lista de horas disponibles para citas
        - n_ants: Número de hormigas (agentes) a usar en cada iteración
        - iterations: Número de iteraciones del algoritmo
        - alpha: Parámetro que controla la importancia de las feromonas
        - beta: Parámetro que controla la importancia de la información heurística
        - rho: Tasa de evaporación de feromonas
        - Q: Constante que afecta la cantidad de feromonas depositadas
        """
        self.graph = graph
        self.fases_orden = fases_orden  
        self.fases_duration = fases_duration  
        self.pacientes = pacientes
        self.medicos = medicos
        self.consultas = consultas
        self.horas = horas
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_solution = None  # Se almacena aquí la mejor solución global
        self.total_costs = []      # Se registran los costes de cada iteración
        self.best_cost = float('inf')  # Se inicializa con valor infinito
        self.execution_time = None  # Se usa para almacenar el tiempo total

    def run(self):
        """
        Ejecuta el algoritmo principal de optimización de colonia de hormigas.
        Retorna la mejor solución encontrada y su coste.
        """
        start_time = time.time()  # Se inicia el cronómetro
        
        for iteration in range(self.iterations):
            ants = [Ant(self.graph, self.fases_orden, self.fases_duration, self.pacientes, 
                        self.alpha, self.beta) for _ in range(self.n_ants)]
            
            # Se inicializan variables para la iteración actual
            iteration_best_cost = float('inf')
            iteration_best_solution = None
            iteration_best_ant = None
            
            # Se procesa cada hormiga de la colonia
            for ant_idx, ant in enumerate(ants):
                # Se limita el número de intentos para evitar bucles infinitos
                max_attempts = len(self.graph.nodes) * 2
                attempts = 0
                
                # Se construye la solución paso a paso
                while attempts < max_attempts and not ant.valid_solution:
                    next_node = ant.choose_next_node()  # Se selecciona el próximo nodo
                    if next_node is None:
                        break  # No quedan nodos válidos
                    ant.move(next_node)  # Se realiza el movimiento
                    attempts += 1
                
                # Se evalúa si la hormiga encontró una solución viable
                if ant.valid_solution:
                    ant.total_cost = self.calcular_coste(ant.visited)
                    # Se actualiza la mejor solución de esta iteración si procede
                    if ant.total_cost < iteration_best_cost:
                        iteration_best_cost = ant.total_cost
                        iteration_best_solution = ant.visited.copy()
                        iteration_best_ant = ant
            
            # Se aplica búsqueda local para refinar la solución
            if iteration_best_solution is not None:
                # Se realizan hasta 3 mejoras incrementales
                current_solution = iteration_best_solution
                current_cost = iteration_best_cost
                
                for _ in range(3):  # Se intenta mejorar hasta 3 veces
                    improved_solution = self.local_search(current_solution)
                    improved_cost = self.calcular_coste(improved_solution)
                    
                    # Se acepta la mejora solo si es significativa (>1%)
                    if improved_cost < current_cost * 0.99:
                        current_solution = improved_solution
                        current_cost = improved_cost
                    else:
                        break  # Se detiene si no hay mejora considerable
                
                # Se actualiza la solución global si es mejor
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_solution = current_solution.copy()
                
                if current_cost < iteration_best_cost:
                    iteration_best_cost = current_cost
                    iteration_best_ant.visited = current_solution.copy()
            
            # Se actualizan las feromonas solo con la mejor hormiga
            if iteration_best_ant:
                self.graph.update_pheromone([iteration_best_ant], self.rho, self.Q)
            
            # Se registra el progreso para análisis posterior
            self.total_costs.append(iteration_best_cost)
        
        # Se calcula el tiempo total de ejecución
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        return self.best_solution, self.best_cost

    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        """
        Calcula el coste de una solución sumando penalizaciones por:
        - Conflictos de recursos (mismo médico o consulta a la misma hora)
        - Secuencia incorrecta de fases para pacientes
        - Tiempos de espera excesivos entre fases
        
        Retorna un valor numérico que representa el coste total.
        """
        # Se preprocesan las asignaciones para optimizar el cálculo
        fases_activas = []
        tiempos_pacientes = defaultdict(list)
        
        # Se usa caché para evitar conversiones repetidas
        hora_cache = {}
        
        # Se procesa cada tupla de asignación
        for asignacion in asignaciones:
            paciente, consulta, hora_str, medico, fase = asignacion
            
            # Se convierte la hora a minutos usando caché
            if hora_str not in hora_cache:
                hora_inicio = datetime.datetime.strptime(hora_str, "%H:%M").time()
                inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
                hora_cache[hora_str] = inicio_min
            else:
                inicio_min = hora_cache[hora_str]
                
            fin_min = inicio_min + self.fases_duration[fase]
            
            # Se almacenan los datos procesados
            fases_activas.append({
                'paciente': paciente,
                'consulta': consulta,
                'fase': fase,
                'inicio': inicio_min,
                'fin': fin_min,
                'medico': medico,
                'orden': self.fases_orden[fase]
            })
            
            # Se registran los tiempos por paciente
            tiempos_pacientes[paciente].append((
                self.fases_orden[fase],
                inicio_min,
                fin_min
            ))
        
        penalty = 0  # Se inicia sin penalizaciones
        
        # Se verifica el uso de recursos mediante indexación espacial
        medico_slots = defaultdict(list)
        consulta_slots = defaultdict(list)
        
        # Se mapea cada recurso a sus franjas de uso
        for fase in fases_activas:
            medico_slots[(fase['medico'], fase['inicio'], fase['fin'])].append(fase)
            consulta_slots[(fase['consulta'], fase['inicio'], fase['fin'])].append(fase)
        
        # Se calculan las penalizaciones por superposición
        for slots in [medico_slots, consulta_slots]:
            for resource_time, fases in slots.items():
                if len(fases) > 1:  # Se detecta conflicto de recursos
                    # Se penaliza cada superposición
                    for i in range(len(fases)):
                        for j in range(i+1, len(fases)):
                            a, b = fases[i], fases[j]
                            overlap_time = min(a['fin'], b['fin']) - max(a['inicio'], b['inicio'])
                            if overlap_time > 0:
                                penalty += 2000 * overlap_time  # Se penaliza proporcionalmente
        
        # Se interrumpe si ya se superó un umbral crítico
        if penalty > 50000:
            return penalty
        
        # Se verifican secuencias de fases por paciente
        for paciente, tiempos in tiempos_pacientes.items():
            # Se ordenan por secuencia lógica
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])
            
            ultima_fase = None
            orden_esperado = 1
            
            for fase_data in tiempos_ordenados:
                orden_actual = fase_data[0]
                inicio_actual = fase_data[1]
                fin_actual = fase_data[2]
                
                # Se verifica el orden correcto de fases
                if orden_actual != orden_esperado:
                    penalty += 10000  # Se penaliza severamente
                    break  # Se termina con este paciente
                
                # Se verifica la coherencia temporal
                if ultima_fase is not None:
                    _, _, fin_prev = ultima_fase
                    
                    if inicio_actual < fin_prev:
                        # Se detecta solapamiento entre fases
                        overlap = fin_prev - inicio_actual
                        penalty += 5000 * overlap  # Se penaliza gravemente
                    else:
                        # Se penaliza por tiempo de espera
                        tiempo_espera = inicio_actual - fin_prev
                        if tiempo_espera > 120:  # Más de 2 horas
                            penalty += tiempo_espera * 2  # Se penaliza con mayor peso
                        else:
                            penalty += tiempo_espera  # Penalización estándar
                
                ultima_fase = fase_data
                orden_esperado += 1
                
                # Se termina si ya es una solución demasiado mala
                if penalty > 100000:
                    return penalty
        
        return penalty
    
    def local_search(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Búsqueda local optimizada que detecta conflictos rápidamente 
        y aplica soluciones específicas.
        """
        original_cost = self.calcular_coste(solution)
        
        # Se omite la búsqueda si la solución ya es buena
        if original_cost < 800:
            return solution
        
        # Se identifican los conflictos existentes
        resource_conflicts, phase_conflicts = self.find_all_conflicts(solution)
        
        # Se intentan mejoras aleatorias si no hay conflictos
        if not resource_conflicts and not phase_conflicts:
            return self._try_random_improvements(solution, original_cost)
        
        # Se abordan primero los conflictos de fase por ser más críticos
        if phase_conflicts:
            for conflict in phase_conflicts[:min(1, len(phase_conflicts))]:
                improved = self.fix_conflicts(solution, conflict, original_cost, True)
                new_cost = self.calcular_coste(improved)
                if new_cost < original_cost:
                    return improved
        
        # Se resuelven después los conflictos de recursos
        if resource_conflicts:
            conflict_counts = {}
            for conflict in resource_conflicts:
                key = (conflict[1], conflict[3])  # Se cuenta por (consulta, médico)
                conflict_counts[key] = conflict_counts.get(key, 0) + 1
            
            # Se ordenan los conflictos por frecuencia
            sorted_conflicts = sorted(
                resource_conflicts,
                key=lambda x: conflict_counts.get((x[1], x[3]), 0),
                reverse=True
            )
            # Se intenta resolver primero los conflictos más frecuentes
            for conflict in sorted_conflicts[:min(1, len(sorted_conflicts))]:
                improved = self.fix_conflicts(solution, conflict, original_cost, False)
                new_cost = self.calcular_coste(improved)
                if new_cost < original_cost:
                    return improved
        
        # Se devuelve la solución original si no se encontraron mejoras
        return solution

    def find_all_conflicts(self, solution):
        """
        Detección rápida de conflictos de recursos y fases en una sola pasada.
        Retorna dos listas: conflictos de recursos y conflictos de fases.
        """
        resource_conflicts = []
        phase_conflicts = []
        
        # Se preprocesan los datos para análisis eficiente
        solution_data = []
        time_cache = {}
        
        # Se agrupan las fases por paciente
        patients_phases = defaultdict(list)
        
        # Se rastrean los usos de recursos
        medicos_uso = defaultdict(list)
        consultas_uso = defaultdict(list)
        
        # Se procesan todas las asignaciones una sola vez
        for asig in solution:
            paciente, consulta, hora_str, medico, fase = asig
            
            # Se convierte el tiempo usando caché
            if hora_str not in time_cache:
                hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
                inicio_mins = hora_obj.hour * 60 + hora_obj.minute
                time_cache[hora_str] = inicio_mins
            else:
                inicio_mins = time_cache[hora_str]
                
            fin_mins = inicio_mins + self.fases_duration[fase]
            
            # Se almacenan los datos normalizados
            asig_data = {
                'original': asig,
                'paciente': paciente,
                'consulta': consulta,
                'medico': medico,
                'fase': fase,
                'inicio_mins': inicio_mins,
                'fin_mins': fin_mins,
                'orden': self.fases_orden[fase]
            }
            solution_data.append(asig_data)
            
            # Se registran los datos por paciente
            patients_phases[paciente].append(asig_data)
            
            # Se registra el uso de recursos
            medicos_uso[medico].append(asig_data)
            consultas_uso[consulta].append(asig_data)
        
        # Se detectan conflictos de recursos
        for resource_name, usage_list in list(medicos_uso.items()) + list(consultas_uso.items()):
            # Se omiten recursos con una sola asignación
            if len(usage_list) <= 1:
                continue
                
            # Se ordenan por tiempo para detección eficiente
            usage_list.sort(key=lambda x: x['inicio_mins'])
            
            # Se buscan superposiciones en un solo recorrido
            for i in range(len(usage_list) - 1):
                current = usage_list[i]
                for j in range(i+1, len(usage_list)):
                    next_usage = usage_list[j]
                    
                    # Se verifica si existe solapamiento
                    if next_usage['inicio_mins'] < current['fin_mins']:
                        resource_conflicts.append(current['original'])
                        resource_conflicts.append(next_usage['original'])
                    else:
                        # Se aprovecha el ordenamiento para salir antes
                        break
        
        # Se detectan conflictos de fase entre pacientes
        for paciente, fases in patients_phases.items():
            # Se ordenan por orden lógico de fase
            fases.sort(key=lambda x: x['orden'])
            
            # Se verifican secuencia y tiempos
            for i in range(len(fases) - 1):
                current = fases[i]
                next_phase = fases[i+1]
                
                # Se verifica orden secuencial correcto
                if next_phase['orden'] != current['orden'] + 1:
                    phase_conflicts.append(current['original'])
                    continue
                    
                # Se verifica coherencia temporal
                if next_phase['inicio_mins'] < current['fin_mins']:
                    phase_conflicts.append(next_phase['original'])
        
        # Se eliminan duplicados para mayor eficiencia
        resource_conflicts = list(set(resource_conflicts))
        phase_conflicts = list(set(phase_conflicts))
        
        return resource_conflicts, phase_conflicts

    def fix_conflicts(self, solution, conflict_asig, original_cost, is_phase):
        """
        Resolución optimizada de conflictos dirigida a problemas específicos.
        Intenta resolver un conflicto concreto probando diferentes cambios.
        """
        paciente, consulta, hora_str, medico, fase = conflict_asig
        best_solution = solution.copy()
        
        # Se convierte la hora una sola vez
        hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
        conflicto_mins = hora_obj.hour * 60 + hora_obj.minute
        
        # Se localiza el conflicto en la solución
        idx = solution.index(conflict_asig)
        
        # Se definen estrategias según el tipo de conflicto
        strategies = []
        
        if is_phase:
            # Se abordan conflictos de fase mediante ajustes temporales
            # Se busca el tiempo de finalización de fase previa
            prev_phase_end = self._get_previous_phase_end(solution, paciente, fase)
            
            # Se generan horas candidatas posteriores a la fase previa
            candidate_times = []
            for hora in self.horas:
                h_obj = datetime.datetime.strptime(hora, "%H:%M")
                h_mins = h_obj.hour * 60 + h_obj.minute
                if h_mins >= prev_phase_end:
                    candidate_times.append(hora)
            
            # Se evalúa cada tiempo candidato
            for nueva_hora in candidate_times:
                new_asig = (paciente, consulta, nueva_hora, medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
        else:
            # Se abordan conflictos de recursos con varias estrategias
            
            # Estrategia 1: Se cambia el médico asignado
            for nuevo_medico in self.medicos:
                if nuevo_medico == medico:
                    continue
                    
                # Se prueba el cambio de médico
                new_asig = (paciente, consulta, hora_str, nuevo_medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
            
            # Estrategia 2: Se cambia la sala de consulta
            for nueva_consulta in self.consultas:
                if nueva_consulta == consulta:
                    continue
                    
                # Se prueba otra sala de consulta
                new_asig = (paciente, nueva_consulta, hora_str, medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
            
            # Estrategia 3: Se modifica el horario
            for nueva_hora in self.horas:
                if nueva_hora == hora_str:
                    continue
                    
                new_asig = (paciente, consulta, nueva_hora, medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
        
        # Se mantiene la solución original si no hay mejoras
        return solution

    def _get_previous_phase_end(self, solution, paciente, current_fase):
        """
        Encuentra el tiempo de finalización de la fase anterior para un paciente.
        Importante para programar fases secuenciales sin solapamiento.
        """
        current_orden = self.fases_orden[current_fase]
        if current_orden == 1:
            return 0  # No hay fase anterior para la primera fase
        
        prev_orden = current_orden - 1
        
        # Se busca la fase previa del paciente
        for asig in solution:
            asig_paciente, _, hora_str, _, asig_fase = asig
            if (asig_paciente == paciente and 
                self.fases_orden[asig_fase] == prev_orden):
                
                # Se calcula el tiempo de finalización
                hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
                inicio_mins = hora_obj.hour * 60 + hora_obj.minute
                return inicio_mins + self.fases_duration[asig_fase]
        
        return 0  # Se retorna 0 si no se encontró fase previa

    def _try_random_improvements(self, solution, original_cost):
        """
        Intenta mejoras aleatorias pequeñas cuando no se encuentran conflictos claros.
        Prueba cambios aleatorios de hora, médico o sala de consulta.
        """
        # Se realizan varios intentos de mejora aleatoria
        for _ in range(3):
            if not solution:
                return solution
                
            # Se selecciona una asignación al azar
            random_idx = random.randint(0, len(solution)-1)
            random_asig = solution[random_idx]
            paciente, consulta, hora_str, medico, fase = random_asig
            
            # Se elige un tipo de cambio aleatorio
            change_type = random.choice(['time', 'doctor', 'room'])
            
            new_solution = solution.copy()
            
            if change_type == 'time':
                # Se prueba un horario diferente
                nueva_hora = random.choice(self.horas)
                new_asig = (paciente, consulta, nueva_hora, medico, fase)
            elif change_type == 'doctor':
                # Se prueba otro médico
                nuevo_medico = random.choice(self.medicos)
                new_asig = (paciente, consulta, hora_str, nuevo_medico, fase)
            else: 
                # Se prueba otra consulta
                nueva_consulta = random.choice(self.consultas)
                new_asig = (paciente, nueva_consulta, hora_str, medico, fase)
            
            new_solution[random_idx] = new_asig
            
            # Se evalúa si la modificación mejoró la solución
            new_cost = self.calcular_coste(new_solution)
            if new_cost < original_cost:
                return new_solution
        
        # Se mantiene la solución original si no se encontraron mejoras
        return solution

    def _generar_hora_respetando_duracion(self, horas: List[str], duracion: int) -> str:
        """
        Genera una hora aleatoria de los slots disponibles, respetando la duración de la fase.
        Asegura que la fase quepa dentro de las horas disponibles y se alinee con slots válidos.
        """
        # Se convierten las horas a datetime
        formato = "%H:%M"
        tiempos = [datetime.datetime.strptime(h, formato) for h in horas]
        
        # Se buscan horas de inicio que permitan completar la fase
        valid_starts = []
        
        for tiempo in tiempos:
            start_mins = tiempo.hour * 60 + tiempo.minute
            end_mins = start_mins + duracion
            
            # Se verifica que la fase se complete dentro del rango horario
            max_mins = max(t.hour * 60 + t.minute for t in tiempos)
            if end_mins <= max_mins:
                valid_starts.append(tiempo)
        
        if not valid_starts:
            # Se usa la hora más temprana si no hay slots válidos
            return min(tiempos).strftime(formato)
        
        # Se elige un tiempo aleatorio entre los válidos
        selected_time = random.choice(valid_starts)
        return selected_time.strftime(formato)

    def plot_convergence(self):
        """
        Genera un gráfico que muestra la evolución del coste a lo largo de las iteraciones.
        Útil para analizar el rendimiento del algoritmo.
        """
        plt.plot(self.total_costs)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.title('Convergencia del ACO')
        
        # Se crea el directorio si no existe
        os.makedirs("/app/plots", exist_ok=True)
        
        # Se guarda la gráfica
        plt.savefig("/app/plots/convergencia.png")
        plt.close()  # Se libera la memoria

    def get_execution_time(self):
        """Devuelve el tiempo de ejecución total en segundos."""
        return self.execution_time
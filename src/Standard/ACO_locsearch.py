import matplotlib.pyplot as plt 
from Standard.Graph import Graph
from utils.Ant import Ant
from typing import Dict,List,Tuple
from collections import defaultdict
import datetime
import random
import time 
import os
import math

class ACO:
    def __init__(self, graph: Graph,  fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str],medicos: List[str],consultas: List[str],horas: List[str], n_ants: int = 10, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1, Q: float = 1.0):
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
        self.best_solution = None
        self.total_costs = []
        self.best_cost = float('inf')
        self.execution_time = None

    def run(self):
        start_time = time.time()
        
        for _ in range(self.iterations):
            ants = [Ant(self.graph, self.fases_orden, self.fases_duration, self.pacientes, 
                        self.alpha, self.beta) for _ in range(self.n_ants)]
            
            # Paso 1: Todas las hormigas construyen soluciones
            iteration_best_cost = float('inf')
            iteration_best_solution = None
            iteration_best_ant = None
            
            for ant in ants:
                while True:
                    next_node = ant.choose_next_node()
                    if next_node is None or ant.valid_solution:
                        break
                    ant.move(next_node)
                
                if ant.valid_solution:
                    ant.total_cost = self.calcular_coste(ant.visited)
                    # Identificar la mejor hormiga de la iteración
                    if ant.total_cost < iteration_best_cost:
                        iteration_best_cost = ant.total_cost
                        iteration_best_solution = ant.visited.copy()
                        iteration_best_ant = ant
                else:
                    ant.total_cost = float('inf')
            
            # Paso 2: Aplicar Local Search solo a la mejor hormiga de la iteración
            if iteration_best_solution is not None:
                improved_solution = self.local_search(iteration_best_solution)
                improved_cost = self.calcular_coste(improved_solution)
                
                # Actualizar la mejor solucion global si hay mejora
                if improved_cost < self.best_cost:
                    self.best_cost = improved_cost
                    self.best_solution = improved_solution.copy()
                    iteration_best_ant.visited = improved_solution.copy()
                    iteration_best_ant.total_cost = improved_cost
            self.total_costs.append(self.best_cost)
            self.graph.update_pheromone(ants, self.rho, self.Q)
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        return self.best_solution, self.best_cost

    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        """
        Calcula el coste total (penalización) de una asignación de horarios con penalizaciones proporcionales:
        1. Solapamientos de médicos/consultas: proporcional al tiempo solapado
        2. Orden correcto de fases por paciente: penalización fija pero severa
        3. Tiempos entre fases consecutivas: proporcional al tiempo de espera
        """
        # Diccionario para almacenar tiempos de cada paciente
        tiempos_pacientes = defaultdict(list)
        penalty = 0
        fases_activas = []  # Almacena todas las fases programadas con sus detalles

        # Procesar todas las asignaciones para extraer tiempos y metadatos
        for asignacion in asignaciones:
            paciente, consulta, hora_str, medico, fase = asignacion
            
            # Convertir hora de string a minutos
            hora_inicio = datetime.datetime.strptime(hora_str, "%H:%M").time()
            inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
            fin_min = inicio_min + self.fases_duration[fase]  # Calcular hora final
            
            # Registrar fase en lista global
            fases_activas.append({
                'paciente': paciente,
                'consulta': consulta,
                'fase': fase,
                'inicio': inicio_min,
                'fin': fin_min,
                'medico': medico,
                'orden': self.fases_orden[fase]  # Orden secuencial de la fase
            })
            
            # Almacenar tiempos por paciente para validaciones posteriores
            tiempos_pacientes[paciente].append((
                self.fases_orden[fase],  # Orden de fase
                inicio_min,              # Minuto de inicio
                fin_min                  # Minuto de finalización
            ))

        # Verificar solapamientos de recursos (médicos y consultas)
        for i in range(len(fases_activas)):
            for j in range(i + 1, len(fases_activas)):
                a = fases_activas[i]
                b = fases_activas[j]
                
                # Calcular solapamiento temporal si existe
                if a['inicio'] < b['fin'] and b['inicio'] < a['fin']:
                    overlap_time = min(a['fin'], b['fin']) - max(a['inicio'], b['inicio'])
                    
                    # Penalizar si mismo médico tiene solapamiento temporal - proporcional al tiempo solapado
                    if a['medico'] == b['medico']:
                        penalty += 2000 * overlap_time  # Base alta × tiempo solapado en minutos
                    
                    # Penalizar si misma consulta tiene solapamiento temporal - proporcional al tiempo solapado
                    if a['consulta'] == b['consulta']:
                        penalty += 2000 * overlap_time  # Base alta × tiempo solapado en minutos

        # Validar secuencia de fases por paciente
        for paciente, tiempos in tiempos_pacientes.items():
            # Ordenar fases por su orden secuencial esperado
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])
            
            # Verificar orden correcto de fases y calculando penalizaciones por espera
            ultima_fase = None
            orden_esperado = 1
            
            for fase_data in tiempos_ordenados:
                orden_actual = fase_data[0]
                inicio_actual = fase_data[1]
                fin_actual = fase_data[2]
                
                # Verificar orden correcto (esta es una restricción fuerte)
                if orden_actual != orden_esperado:
                    # Penalización severa por orden incorrecto
                    penalty += 10000
                    break  # Solo contabilizar una vez por paciente
                
                # Verificar continuidad temporal entre fases
                if ultima_fase is not None:
                    orden_prev, _, fin_prev = ultima_fase
                    
                    # Si hay solapamiento temporal entre fases del mismo paciente
                    if inicio_actual < fin_prev:
                        overlap = fin_prev - inicio_actual
                        penalty += 5000 * overlap  # Penalización proporcional por solapamiento
                    else:
                        # Penalización menor por tiempo de espera excesivo
                        tiempo_espera = inicio_actual - fin_prev
                        # Penalizar más cuando el tiempo de espera es excesivo
                        if tiempo_espera > 120:  # Más de 2 horas de espera
                            penalty += (tiempo_espera - 120) * 2  # Penalización creciente
                        elif tiempo_espera > 60:  # Más de 1 hora de espera
                            penalty += (tiempo_espera - 60)  # Penalización moderada
                        else:
                            penalty += tiempo_espera * 0.5  # Penalización leve
                
                ultima_fase = fase_data
                orden_esperado += 1

        return penalty
    
    def local_search(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Búsqueda local mejorada que:
        1. Prioriza resolver los conflictos más graves primero
        2. Realiza múltiples intentos con diferentes enfoques
        3. Utiliza un enfoque de enfriamiento simulado para aceptar ocasionalmente soluciones peores
        """
        original_cost = self.calcular_coste(solution)
        best_solution = solution.copy()
        best_cost = original_cost
        
        # Número de intentos de mejora
        max_iterations = 1
        # Temperatura para enfriamiento simulado
        temperature = 100.0
        cooling_rate = 0.95
        
        for iteration in range(max_iterations):
            # Detectar todos los conflictos
            phase_conflicts = self._find_phase_conflicts(solution)
            resource_conflicts = self._find_resource_conflicts(solution)
            
            # Si no hay conflictos, terminar la búsqueda
            if not phase_conflicts and not resource_conflicts:
                break
            
            # Determinar qué tipo de conflicto resolver
            conflict_type = None
            if phase_conflicts and resource_conflicts:
                # Priorizar fase o recurso según gravedad (tamaño de listas)
                if len(phase_conflicts) > len(resource_conflicts):
                    conflict_type = "phase"
                    conflicts = phase_conflicts
                else:
                    conflict_type = "resource"
                    conflicts = resource_conflicts
            elif phase_conflicts:
                conflict_type = "phase"
                conflicts = phase_conflicts
            else:
                conflict_type = "resource"
                conflicts = resource_conflicts
            
            # Seleccionar un conflicto a resolver - priorizar los más problemáticos
            # (aquí podríamos calcular 'conflictividad' de cada asignación)
            conflict_counts = {}
            for conflict in conflicts:
                paciente, consulta, hora, medico, fase = conflict
                key = (paciente, fase)
                if key not in conflict_counts:
                    conflict_counts[key] = 0
                conflict_counts[key] += 1
                
            # Ordenar conflictos por número de apariciones (más frecuentes = más problemáticos)
            sorted_conflicts = sorted(conflicts, 
                                    key=lambda x: conflict_counts.get((x[0], x[4]), 0), 
                                    reverse=True)
            
            # Intentar resolver el conflicto más grave
            if sorted_conflicts:
                conflict = sorted_conflicts[0]
            
            # Modificar la solución candidata, intentando resolver el conflicto.
            modified = self._generate_candidate(solution.copy(), conflict, conflict_type == "phase")
            cost = self.calcular_coste(modified)
            
            # Decidir si aceptamos esta solución (siempre si mejora, o probabilísticamente si empeora)
            if cost < best_cost:
                best_solution = modified
                best_cost = cost
                solution = modified  # Continuar desde esta solución
            else:
                # Enfriamiento simulado: aceptar soluciones peores con cierta probabilidad
                delta = cost - best_cost
                probability = math.exp(-delta / temperature)
                if random.random() < probability:
                    solution = modified  # Aceptar solución peor para explorar
            
            # Reducir la temperatura
            temperature *= cooling_rate
        
        return best_solution

    def _find_phase_conflicts(self, solution):
        conflicts = []
        pacientes = defaultdict(list)
        
        # Agrupar asignaciones por paciente
        for asig in solution:
            pacientes[asig[0]].append(asig)  # asig[0] = nombre del paciente
        
        # Iterar sobre cada paciente y sus asignaciones
        for paciente, lista_asignaciones in pacientes.items():  # ¡Clave del error!
            try:
                # Ordenar por orden de fases
                asignaciones_ordenadas = sorted(
                    lista_asignaciones,  # Usar la lista de asignaciones, no el tuple
                    key=lambda x: self.fases_orden[x[4]]
                )
                
                last_end = 0
                for asig in asignaciones_ordenadas:
                    # Convertir hora a minutos desde medianoche
                    hora_obj = datetime.datetime.strptime(asig[2], "%H:%M")
                    inicio = hora_obj.hour * 60 + hora_obj.minute
                    duracion = self.fases_duration[asig[4]]
                    fin = inicio + duracion
                    
                    # Detectar solapamiento solo si no es la primera fase
                    if self.fases_orden[asig[4]] > 1 and inicio < last_end:
                        conflicts.append(asig)
                    last_end = fin
                    
            except KeyError as e:
                print(f"¡Fase no registrada! Paciente: {paciente}, Fase: {e}")
                print("Fases válidas:", self.fases_orden.keys())
                raise
        
        return conflicts

    def _find_resource_conflicts(self, solution):
        conflicts = []
        time_slots = defaultdict(set)
        
        for idx, asig in enumerate(solution):
            key = (asig[2], asig[3], asig[1])  # (hora, médico, consulta)
            
            # Si ya existen índices para esta clave, todos son conflictivos
            if key in time_slots and len(time_slots[key]) > 0:
                # Añadir todos los índices previos y el actual al conflicto
                for existing_idx in time_slots[key]:
                    conflicts.append(solution[existing_idx])
                conflicts.append(asig)
            
            time_slots[key].add(idx)
        
        # Eliminar duplicados usando un set
        return list({tuple(x) for x in conflicts})

    def _generate_candidate(self, solution: List[Tuple], conflict_asig: Tuple, is_phase: bool) -> List[Tuple]:
        """
        Generates a candidate solution by trying to resolve a specific conflict.
        Uses more sophisticated strategies based on the type of conflict.
        Properly respects phase durations when scheduling.
        """
        paciente, consulta_orig, hora_orig, medico_orig, fase = conflict_asig
        idx = solution.index(conflict_asig)
        duracion_fase = self.fases_duration[fase]  # Get the duration of this phase
        
        # Resolution strategies based on conflict type
        if is_phase:
            # For phase conflicts, prioritize changes that maintain the correct order
            estrategia = random.choice(["cambiar_hora", "cambiar_recursos"])
            
            if estrategia == "cambiar_hora":
                # Try to find a viable schedule that maintains the phase order
                otras_fases = [s for s in solution if s[0] == paciente and s[4] != fase]
                otras_fases_info = []
                
                for otra in otras_fases:
                    orden = self.fases_orden[otra[4]]
                    hora = datetime.datetime.strptime(otra[2], "%H:%M")
                    minutos = hora.hour * 60 + hora.minute
                    duracion = self.fases_duration[otra[4]]
                    otras_fases_info.append((orden, minutos, minutos + duracion, otra))
                
                # Sort by phase order
                otras_fases_info.sort()
                
                # Determine available time windows based on this phase's order
                orden_actual = self.fases_orden[fase]
                antes = [f for f in otras_fases_info if f[0] < orden_actual]
                despues = [f for f in otras_fases_info if f[0] > orden_actual]
                
                # Calculate limits
                limite_inferior = max([f[2] for f in antes], default=0)  # End of previous phase
                limite_superior = min([f[1] for f in despues], default=24*60)  # Start of next phase
                
                # Generate time within limits, respecting phase duration
                if limite_superior > limite_inferior + duracion_fase:
                    # There's space, generate viable time
                    minutos_inicio = random.randint(limite_inferior, limite_superior - duracion_fase)
                    horas = minutos_inicio // 60
                    minutos = minutos_inicio % 60
                    nueva_hora = f"{horas:02d}:{minutos:02d}"
                else:
                    # No ideal space, try to minimize conflict
                    # Generate a time that respects available slots and phase duration
                    nueva_hora = self._generar_hora_respetando_duracion(self.horas, duracion_fase)
                
                nueva_asig = (paciente, consulta_orig, nueva_hora, medico_orig, fase)
                
            else:  # cambiar_recursos
                # Maintain time but change doctor and/or consultation
                nueva_hora = hora_orig
                hora_inicio_mins = self._hora_a_minutos(hora_orig)
                hora_fin_mins = hora_inicio_mins + duracion_fase
                
                # Identify resources occupied during this time span (considering duration)
                ocupados_medicos = set()
                ocupadas_consultas = set()
                
                for s in solution:
                    if s == conflict_asig:
                        continue
                    
                    s_inicio = self._hora_a_minutos(s[2])
                    s_fin = s_inicio + self.fases_duration[s[4]]
                    
                    # Check for any overlap
                    if s_inicio < hora_fin_mins and s_fin > hora_inicio_mins:
                        ocupados_medicos.add(s[3])
                        ocupadas_consultas.add(s[1])
                
                # Filter available resources
                medicos_disponibles = [m for m in self.medicos if m not in ocupados_medicos]
                consultas_disponibles = [c for c in self.consultas if c not in ocupadas_consultas]
                
                # Select new values
                nuevo_medico = random.choice(medicos_disponibles) if medicos_disponibles else medico_orig
                nueva_consulta = random.choice(consultas_disponibles) if consultas_disponibles else consulta_orig
                
                nueva_asig = (paciente, nueva_consulta, nueva_hora, nuevo_medico, fase)
        
        else:  # Resource conflict
            # For resource conflicts, try several strategies
            estrategias = ["cambiar_hora", "cambiar_medico", "cambiar_consulta", "cambiar_ambos"]
            estrategia = random.choice(estrategias)
            
            if estrategia == "cambiar_hora":
                # Generate new time avoiding conflicts, considering phase duration
                hora_inicial_mins = self._hora_a_minutos(hora_orig)
                
                # Find times that would conflict with this phase's duration
                conflictos_tiempo = set()
                for s in solution:
                    if s == conflict_asig:
                        continue
                    
                    if s[3] == medico_orig or s[1] == consulta_orig:  # Same doctor or consultation
                        s_inicio = self._hora_a_minutos(s[2])
                        s_fin = s_inicio + self.fases_duration[s[4]]
                        
                        # Add all potential start times that would conflict
                        for t in range(max(0, s_inicio - duracion_fase + 1), s_fin):
                            h = t // 60
                            m = t % 60
                            conflictos_tiempo.add(f"{h:02d}:{m:02d}")
                
                # Filter available hours
                horas_disponibles = [h for h in self.horas if h not in conflictos_tiempo]
                
                if horas_disponibles:
                    nueva_hora = random.choice(horas_disponibles)
                else:
                    # If all are occupied, try with smaller intervals
                    nueva_hora = self._generar_hora_respetando_duracion(self.horas, duracion_fase)
                
                nueva_asig = (paciente, consulta_orig, nueva_hora, medico_orig, fase)
                
            elif estrategia == "cambiar_medico":
                nueva_hora = hora_orig
                hora_inicio_mins = self._hora_a_minutos(hora_orig)
                hora_fin_mins = hora_inicio_mins + duracion_fase
                
                # Find doctors with conflicts during this time span
                ocupados_medicos = set()
                for s in solution:
                    if s == conflict_asig:
                        continue
                    
                    s_inicio = self._hora_a_minutos(s[2])
                    s_fin = s_inicio + self.fases_duration[s[4]]
                    
                    # Check for any overlap
                    if s_inicio < hora_fin_mins and s_fin > hora_inicio_mins:
                        ocupados_medicos.add(s[3])
                
                medicos_disponibles = [m for m in self.medicos if m not in ocupados_medicos]
                
                nuevo_medico = random.choice(medicos_disponibles) if medicos_disponibles else medico_orig
                nueva_asig = (paciente, consulta_orig, nueva_hora, nuevo_medico, fase)
                
            elif estrategia == "cambiar_consulta":
                nueva_hora = hora_orig
                hora_inicio_mins = self._hora_a_minutos(hora_orig)
                hora_fin_mins = hora_inicio_mins + duracion_fase
                
                # Find consultations with conflicts during this time span
                ocupadas_consultas = set()
                for s in solution:
                    if s == conflict_asig:
                        continue
                    
                    s_inicio = self._hora_a_minutos(s[2])
                    s_fin = s_inicio + self.fases_duration[s[4]]
                    
                    # Check for any overlap
                    if s_inicio < hora_fin_mins and s_fin > hora_inicio_mins:
                        ocupadas_consultas.add(s[1])
                
                consultas_disponibles = [c for c in self.consultas if c not in ocupadas_consultas]
                
                nueva_consulta = random.choice(consultas_disponibles) if consultas_disponibles else consulta_orig
                nueva_asig = (paciente, nueva_consulta, nueva_hora, medico_orig, fase)
                
            else:  # cambiar_ambos
                nueva_hora = hora_orig
                hora_inicio_mins = self._hora_a_minutos(hora_orig)
                hora_fin_mins = hora_inicio_mins + duracion_fase
                
                # Find resources with conflicts during this time span
                ocupados_medicos = set()
                ocupadas_consultas = set()
                
                for s in solution:
                    if s == conflict_asig:
                        continue
                    
                    s_inicio = self._hora_a_minutos(s[2])
                    s_fin = s_inicio + self.fases_duration[s[4]]
                    
                    # Check for any overlap
                    if s_inicio < hora_fin_mins and s_fin > hora_inicio_mins:
                        ocupados_medicos.add(s[3])
                        ocupadas_consultas.add(s[1])
                
                medicos_disponibles = [m for m in self.medicos if m not in ocupados_medicos]
                consultas_disponibles = [c for c in self.consultas if c not in ocupadas_consultas]
                
                nuevo_medico = random.choice(medicos_disponibles) if medicos_disponibles else medico_orig
                nueva_consulta = random.choice(consultas_disponibles) if consultas_disponibles else consulta_orig
                
                nueva_asig = (paciente, nueva_consulta, nueva_hora, nuevo_medico, fase)
        
        # Update the solution with the new assignment
        solution[idx] = nueva_asig
        return solution

    def _hora_a_minutos(self, hora_str: str) -> int:
        """Convert time string to minutes since midnight"""
        hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
        return hora_obj.hour * 60 + hora_obj.minute

    def _generar_hora_respetando_duracion(self, horas: List[str], duracion: int) -> str:
        """
        Generate a random time from available slots, respecting phase duration.
        Ensures the phase fits within available hours.
        """
        # Convert to datetime objects and find the range
        formato = "%H:%M"
        tiempos = [datetime.datetime.strptime(h, formato) for h in horas]
        min_t = min(tiempos)
        max_t = max(tiempos)
        
        # Convert times to minutes for easier calculation
        min_mins = min_t.hour * 60 + min_t.minute
        max_mins = max_t.hour * 60 + max_t.minute
        
        # Ensure the phase fits within the day
        max_start_mins = max_mins - duracion
        
        if max_start_mins < min_mins:
            # Not enough time in the day, return earliest time
            return min_t.strftime(formato)
        
        # Generate a random start time that allows the phase to complete
        start_mins = random.randint(min_mins, max_start_mins)
        hours = start_mins // 60
        minutes = start_mins % 60
        
        return f"{hours:02d}:{minutes:02d}"
    def plot_convergence(self):
        plt.plot(self.total_costs)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor Distancia')
        plt.title('Convergencia del ACO')
        
        # Crear directorio si no existe
        os.makedirs("/app/plots", exist_ok=True)
        
        # Guardar la imagen
        plt.savefig("/app/plots/convergencia.png")
        plt.close()  # Limpiar la figura

    def get_execution_time(self):
        """Devuelve el tiempo de ejecución total en segundos."""
        return self.execution_time


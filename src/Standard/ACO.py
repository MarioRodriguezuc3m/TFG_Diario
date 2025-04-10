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
        
        for iteration in range(self.iterations):
            # Create ants with shared reference data to reduce memory usage
            ants = [Ant(self.graph, self.fases_orden, self.fases_duration, self.pacientes, 
                        self.alpha, self.beta) for _ in range(self.n_ants)]
            
            # Process ants in parallel or batches if possible
            iteration_best_cost = float('inf')
            iteration_best_solution = None
            iteration_best_ant = None
            
            for ant_idx, ant in enumerate(ants):
                # Early termination if ant isn't making progress
                max_attempts = len(self.graph.nodes) * 2
                attempts = 0
                
                while attempts < max_attempts and not ant.valid_solution:
                    next_node = ant.choose_next_node()
                    if next_node is None:
                        break
                    ant.move(next_node)
                    attempts += 1
                
                if ant.valid_solution:
                    ant.total_cost = self.calcular_coste(ant.visited)
                    if ant.total_cost < iteration_best_cost:
                        iteration_best_cost = ant.total_cost
                        iteration_best_solution = ant.visited.copy()
                        iteration_best_ant = ant
            
            # Apply local search only if we found a valid solution
            if iteration_best_solution is not None:
                # Multiple local search attempts with diminishing returns check
                current_solution = iteration_best_solution
                current_cost = iteration_best_cost
                
                for _ in range(3):  # Try up to 3 sequential improvements
                    improved_solution = self.local_search(current_solution)
                    improved_cost = self.calcular_coste(improved_solution)
                    
                    if improved_cost < current_cost * 0.99:  # At least 1% improvement
                        current_solution = improved_solution
                        current_cost = improved_cost
                    else:
                        break  # Stop if improvements are marginal
                
                # Update best solution if improved
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_solution = current_solution.copy()
            
            # Update pheromones only based on best ant(s)
            if iteration_best_ant:
                self.graph.update_pheromone([iteration_best_ant], self.rho, self.Q)
            
            self.total_costs.append(self.best_cost)
        
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        return self.best_solution, self.best_cost

    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        """Función para calcular el coste de una solución."""
        # Pre-process all assignments for faster lookup
        fases_activas = []
        tiempos_pacientes = defaultdict(list)
        
        # Cache for datetime conversions
        hora_cache = {}
        
        for asignacion in asignaciones:
            paciente, consulta, hora_str, medico, fase = asignacion
            
            # Cache datetime conversion to avoid repeated parsing
            if hora_str not in hora_cache:
                hora_inicio = datetime.datetime.strptime(hora_str, "%H:%M").time()
                inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
                hora_cache[hora_str] = inicio_min
            else:
                inicio_min = hora_cache[hora_str]
                
            fin_min = inicio_min + self.fases_duration[fase]
            
            # Store phase info
            fases_activas.append({
                'paciente': paciente,
                'consulta': consulta,
                'fase': fase,
                'inicio': inicio_min,
                'fin': fin_min,
                'medico': medico,
                'orden': self.fases_orden[fase]
            })
            
            # Store patient times
            tiempos_pacientes[paciente].append((
                self.fases_orden[fase],
                inicio_min,
                fin_min
            ))
        
        penalty = 0
        
        # Check resource conflicts using spatial indexing approach
        # Track resource usage by time slots
        medico_slots = defaultdict(list)
        consulta_slots = defaultdict(list)
        
        for fase in fases_activas:
            medico_slots[(fase['medico'], fase['inicio'], fase['fin'])].append(fase)
            consulta_slots[(fase['consulta'], fase['inicio'], fase['fin'])].append(fase)
        
        # Calculate overlaps more efficiently
        for slots in [medico_slots, consulta_slots]:
            for resource_time, fases in slots.items():
                if len(fases) > 1:  # Conflict detected
                    # Calculate overlap penalty for each pair
                    for i in range(len(fases)):
                        for j in range(i+1, len(fases)):
                            a, b = fases[i], fases[j]
                            overlap_time = min(a['fin'], b['fin']) - max(a['inicio'], b['inicio'])
                            if overlap_time > 0:
                                penalty += 2000 * overlap_time
        
        # Early return if penalty is already too high
        if penalty > 50000:
            return penalty
        
        # Check phase sequences
        for paciente, tiempos in tiempos_pacientes.items():
            # Sort by expected phase order
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])
            
            ultima_fase = None
            orden_esperado = 1
            
            for fase_data in tiempos_ordenados:
                orden_actual = fase_data[0]
                inicio_actual = fase_data[1]
                fin_actual = fase_data[2]
                
                # Check correct order
                if orden_actual != orden_esperado:
                    penalty += 10000
                    break  # Early exit for this patient
                
                # Check temporal continuity
                if ultima_fase is not None:
                    _, _, fin_prev = ultima_fase
                    
                    if inicio_actual < fin_prev:
                        overlap = fin_prev - inicio_actual
                        penalty += 5000 * overlap
                    else:
                        tiempo_espera = inicio_actual - fin_prev
                        if tiempo_espera > 120:
                            penalty += tiempo_espera * 2
                        else:
                            penalty += tiempo_espera
                
                ultima_fase = fase_data
                orden_esperado += 1
                
                # Early return if penalty is too high
                if penalty > 100000:
                    return penalty
        
        return penalty
    
    def local_search(self, solution: List[Tuple]) -> List[Tuple]:
        """
        Optimized local search with faster conflict detection and targeted fixes.
        """
        original_cost = self.calcular_coste(solution)
        
        # Quick check if solution is already good enough
        if original_cost < 800:  # Skip local search for already good solutions
            return solution
        
        # Use efficient conflict detection that avoids repeated time conversions
        resource_conflicts, phase_conflicts = self._find_all_conflicts_fast(solution)
        
        # No conflicts found - try small random perturbations instead
        if not resource_conflicts and not phase_conflicts:
            return self._try_random_improvements(solution, original_cost)
        
        # Focus on most severe conflicts first (typically phase conflicts)
        if phase_conflicts:
            # Try to fix the most problematic resources first (those with most conflicts)
            conflict_counts = {}
            for conflict in phase_conflicts:
                key = (conflict[1], conflict[3])  # (consulta, medico)
                conflict_counts[key] = conflict_counts.get(key, 0) + 1
            
            # Sort conflicts by severity (most conflicts first)
            sorted_conflicts = sorted(
                phase_conflicts,
                key=lambda x: conflict_counts.get((x[1], x[3]), 0),
                reverse=True
            )
            
            # Try fixing each conflict, but limit to the top N most severe
            for conflict in sorted_conflicts[:min(1, len(sorted_conflicts))]:
                improved = self._fix_conflict_fast(solution, conflict, original_cost, False)
                new_cost = self.calcular_coste(improved)
                if new_cost < original_cost:
                    return improved
        
        # Then try fixing resource conflicts
        if resource_conflicts:
            for conflict in resource_conflicts[:min(1, len(resource_conflicts))]:
                improved = self._fix_conflict_fast(solution, conflict, original_cost, True)
                new_cost = self.calcular_coste(improved)
                if new_cost < original_cost:
                    return improved
        
        # If no improvements found, return original solution
        return solution

    def _find_all_conflicts_fast(self, solution):
        """
        Fast detection of both resource and phase conflicts in a single pass.
        """
        resource_conflicts = []
        phase_conflicts = []
        
        # Pre-process time conversions and organize data
        solution_data = []
        time_cache = {}
        
        # Group by patients for phase conflict detection
        patients_phases = defaultdict(list)
        
        # Resource usage tracking
        medicos_uso = defaultdict(list)
        consultas_uso = defaultdict(list)
        
        # Process all assignments in one pass
        for asig in solution:
            paciente, consulta, hora_str, medico, fase = asig
            
            # Convert time to minutes - use cache for efficiency
            if hora_str not in time_cache:
                hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
                inicio_mins = hora_obj.hour * 60 + hora_obj.minute
                time_cache[hora_str] = inicio_mins
            else:
                inicio_mins = time_cache[hora_str]
                
            fin_mins = inicio_mins + self.fases_duration[fase]
            
            # Store processed data
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
            
            # Track patient phases
            patients_phases[paciente].append(asig_data)
            
            # Track resource usage
            medicos_uso[medico].append(asig_data)
            consultas_uso[consulta].append(asig_data)
        
        # Check for resource conflicts
        for resource_name, usage_list in list(medicos_uso.items()) + list(consultas_uso.items()):
            # Only check resources with multiple assignments
            if len(usage_list) <= 1:
                continue
                
            # Sort by start time for efficient overlap checking
            usage_list.sort(key=lambda x: x['inicio_mins'])
            
            # Find overlaps with a single pass
            for i in range(len(usage_list) - 1):
                current = usage_list[i]
                for j in range(i+1, len(usage_list)):
                    next_usage = usage_list[j]
                    
                    # Check if there's overlap
                    if next_usage['inicio_mins'] < current['fin_mins']:
                        resource_conflicts.append(current['original'])
                        resource_conflicts.append(next_usage['original'])
                    else:
                        # No more overlaps since list is sorted
                        break
        
        # Check for phase conflicts
        for paciente, fases in patients_phases.items():
            # Sort by phase order for sequential check
            fases.sort(key=lambda x: x['orden'])
            
            # Check phase sequence and overlaps
            for i in range(len(fases) - 1):
                current = fases[i]
                next_phase = fases[i+1]
                
                # Check for correct order
                if next_phase['orden'] != current['orden'] + 1:
                    phase_conflicts.append(current['original'])
                    continue
                    
                # Check for time sequence
                if next_phase['inicio_mins'] < current['fin_mins']:
                    phase_conflicts.append(next_phase['original'])
        
        # Remove duplicates
        resource_conflicts = list(set(resource_conflicts))
        phase_conflicts = list(set(phase_conflicts))
        
        return resource_conflicts, phase_conflicts

    def _fix_conflict_fast(self, solution, conflict_asig, original_cost, is_phase):
        """
        Optimized conflict resolution that targets specific issues.
        """
        paciente, consulta, hora_str, medico, fase = conflict_asig
        best_solution = solution.copy()
        
        # Convert the conflict time once
        hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
        conflicto_mins = hora_obj.hour * 60 + hora_obj.minute
        
        # Find the index of this conflict in the solution
        idx = solution.index(conflict_asig)
        
        # Strategy selection based on conflict type
        strategies = []
        
        if is_phase:
            # Phase conflicts: focus on time adjustments
            # Find the right time slot after the previous phase
            prev_phase_end = self._get_previous_phase_end(solution, paciente, fase)
            
            # Generate candidate times starting from after the previous phase
            candidate_times = []
            for hora in self.horas:
                h_obj = datetime.datetime.strptime(hora, "%H:%M")
                h_mins = h_obj.hour * 60 + h_obj.minute
                if h_mins >= prev_phase_end:
                    candidate_times.append(hora)
            
            # Try each candidate time
            for nueva_hora in candidate_times:
                new_asig = (paciente, consulta, nueva_hora, medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
        else:
            # Resource conflicts: try different resources or times
            
            # First try: change doctor/room without changing time
            for nuevo_medico in self.medicos:
                if nuevo_medico == medico:
                    continue
                    
                # Try new doctor assignment
                new_asig = (paciente, consulta, hora_str, nuevo_medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
            
            # Second try: change consultation room
            for nueva_consulta in self.consultas:
                if nueva_consulta == consulta:
                    continue
                    
                # Try new consultation room
                new_asig = (paciente, consulta, hora_str, medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
            
            # Last try: move to different time slot
            for nueva_hora in self.horas:
                if nueva_hora == hora_str:
                    continue
                    
                new_asig = (paciente, consulta, nueva_hora, medico, fase)
                new_solution = solution.copy()
                new_solution[idx] = new_asig
                
                new_cost = self.calcular_coste(new_solution)
                if new_cost < original_cost:
                    return new_solution
        
        # If no improvements found, return original solution
        return solution

    def _get_previous_phase_end(self, solution, paciente, current_fase):
        """
        Find the end time of the previous phase for a patient.
        """
        current_orden = self.fases_orden[current_fase]
        if current_orden == 1:
            return 0  # No previous phase
        
        prev_orden = current_orden - 1
        
        # Find the previous phase assignment
        for asig in solution:
            asig_paciente, _, hora_str, _, asig_fase = asig
            if (asig_paciente == paciente and 
                self.fases_orden[asig_fase] == prev_orden):
                
                # Calculate end time of previous phase
                hora_obj = datetime.datetime.strptime(hora_str, "%H:%M")
                inicio_mins = hora_obj.hour * 60 + hora_obj.minute
                return inicio_mins + self.fases_duration[asig_fase]
        
        return 0  # No previous phase found

    def _try_random_improvements(self, solution, original_cost):
        """
        Try random small improvements when no clear conflicts are found.
        """
        # Pick a random assignment to try to improve
        for _ in range(3):  # Try a few random changes
            if not solution:
                return solution
                
            random_idx = random.randint(0, len(solution)-1)
            random_asig = solution[random_idx]
            paciente, consulta, hora_str, medico, fase = random_asig
            
            # Try a different time or resource
            change_type = random.choice(['time', 'doctor', 'room'])
            
            new_solution = solution.copy()
            
            if change_type == 'time':
                nueva_hora = random.choice(self.horas)
                new_asig = (paciente, consulta, nueva_hora, medico, fase)
            elif change_type == 'doctor':
                nuevo_medico = random.choice(self.medicos)
                new_asig = (paciente, consulta, hora_str, nuevo_medico, fase)
            else:  # room
                nueva_consulta = random.choice(self.consultas)
                new_asig = (paciente, nueva_consulta, hora_str, medico, fase)
            
            new_solution[random_idx] = new_asig
            
            new_cost = self.calcular_coste(new_solution)
            if new_cost < original_cost:
                return new_solution
        
        return solution

    def _generar_hora_respetando_duracion(self, horas: List[str], duracion: int) -> str:
        """
        Generate a random time from available slots, respecting phase duration.
        Ensures the phase fits within available hours and aligns with valid time slots.
        """
        # Convert available hours to datetime objects
        formato = "%H:%M"
        tiempos = [datetime.datetime.strptime(h, formato) for h in horas]
        
        # Find valid start times that allow the phase to complete
        valid_starts = []
        
        for tiempo in tiempos:
            start_mins = tiempo.hour * 60 + tiempo.minute
            end_mins = start_mins + duracion
            
            # Check if this time slot can accommodate the phase
            # by ensuring the end time is within the range of available hours
            max_mins = max(t.hour * 60 + t.minute for t in tiempos)
            if end_mins <= max_mins:
                valid_starts.append(tiempo)
        
        if not valid_starts:
            # No valid slots found, return earliest time as fallback
            return min(tiempos).strftime(formato)
        
        # Select a random valid start time
        selected_time = random.choice(valid_starts)
        return selected_time.strftime(formato)

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


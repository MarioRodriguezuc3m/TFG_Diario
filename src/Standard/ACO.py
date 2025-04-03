import matplotlib.pyplot as plt 
from Standard.Graph import Graph
from utils.Ant import Ant
from typing import Dict,List,Tuple
from collections import defaultdict
import datetime
import random
import time 
import os

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
        Calcula el coste total (penalización) de una asignación de horarios según restricciones:
        1. Solapamientos de médicos/consultas
        2. Orden correcto de fases por paciente
        3. Tiempos entre fases consecutivas
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
                
                # Penalizar si mismo médico tiene solapamiento temporal
                if a['medico'] == b['medico'] and (a['inicio'] < b['fin'] and b['inicio'] < a['fin']):
                    penalty += 1000  # Penalización alta por conflicto de médico
                    
                # Penalizar si misma consulta tiene solapamiento temporal
                if a['consulta'] == b['consulta'] and (a['inicio'] < b['fin'] and b['inicio'] < a['fin']):
                    penalty += 1000  # Penalización alta por conflicto de consulta

        # Validar secuencia de fases por paciente
        for paciente, tiempos in tiempos_pacientes.items():
            # Ordenar fases por su orden secuencial esperado
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])  
            orden_esperado = 1
            
            # Verificar orden correcto de fases (1, 2, 3...)
            for fase_data in tiempos_ordenados:
                orden_actual = fase_data[0]
                if orden_actual != orden_esperado:
                    penalty += 2000  # Penalización alta por orden incorrecto
                    break  # Solo contabilizar una vez por paciente
                orden_esperado += 1
            
            # Verificar continuidad temporal entre fases del mismo paciente
            for i in range(1, len(tiempos_ordenados)):
                fase_prev = tiempos_ordenados[i-1]
                fase_actual = tiempos_ordenados[i]
                
                # Penalizar si hay solapamiento entre fases consecutivas
                if fase_actual[1] < fase_prev[2]:
                    penalty += 2000  # Penalización alta por solapamiento interno
                    break
                else:
                    # Añadir tiempo muerto entre fases como penalización menor
                    penalty += fase_actual[1] - fase_prev[2]

        return penalty
    
    def local_search(self, solution: List[Tuple]) -> List[Tuple]:
        original_cost = self.calcular_coste(solution)
        
        # Detectar todos los conflictos
        phase_conflicts = self._find_phase_conflicts(solution)
        resource_conflicts = self._find_resource_conflicts(solution)
        
        # Priorizar conflictos de fase
        if phase_conflicts:
            conflict = random.choice(phase_conflicts)
            return self._fix_conflict(solution, conflict, original_cost, is_phase=True)
        
        if resource_conflicts:
            conflict = random.choice(resource_conflicts)
            return self._fix_conflict(solution, conflict, original_cost, is_phase=False)
        
        return solution  # No hay conflictos

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

    def _fix_conflict(self, solution, conflict_asig, original_cost, is_phase):
        paciente, _, hora_str, _, fase = conflict_asig
        best_solution = solution.copy()
        
        # Obtener todos los médicos y consultas del sistema
        todos_medicos = self.medicos  # Asumiendo que el grafo tiene esta propiedad
        todas_consultas = self.consultas  # Asumiendo que el grafo tiene esta propiedad
        
        for _ in range(20):
            new_solution = solution.copy()
            idx = new_solution.index(conflict_asig)
            nueva_consulta = conflict_asig[1]  # Valor por defecto
            nuevo_medico = conflict_asig[3]  # Valor por defecto
            nueva_hora = hora_str
            
            if is_phase:
                # Cambiar hora para conflictos de fase
                nueva_hora = self.generar_hora_aleatoria(self.horas, 60)
            else:
                if random.random() < 0.5:  # 50% cambiar hora
                    nueva_hora = self.generar_hora_aleatoria(self.horas, 60)
                else:  # 50% cambiar médico/consulta
                    # Obtener recursos ocupados en la hora original del conflicto
                    ocupados_medicos = {asig[3] for asig in solution if asig[2] == hora_str}
                    ocupadas_consultas = {asig[1] for asig in solution if asig[2] == hora_str}
                    
                    # Filtrar disponibles
                    medicos_disponibles = [m for m in todos_medicos if m not in ocupados_medicos]
                    consultas_disponibles = [c for c in todas_consultas if c not in ocupadas_consultas]
                    
                    # Seleccionar nuevos valores solo si hay disponibles
                    if medicos_disponibles:
                        nuevo_medico = random.choice(medicos_disponibles)
                    if consultas_disponibles:
                        nueva_consulta = random.choice(consultas_disponibles)
            
            # Crear nueva asignación
            new_asig = (paciente,nueva_consulta,nueva_hora,nuevo_medico,fase)
            new_solution[idx] = new_asig
            
            new_cost = self.calcular_coste(new_solution)
            if new_cost < original_cost:
                best_solution = new_solution
                break
        
        return best_solution

    @staticmethod
    def generar_hora_aleatoria(horas: List[str], intervalo: int) -> str:
        # Convertir a objetos datetime y encontrar el rango
        formato = "%H:%M"
        tiempos = [datetime.datetime.strptime(h, formato) for h in horas]
        min_t = min(tiempos)
        max_t = max(tiempos)
        
        # Generar todos los intervalos posibles
        slots = []
        current = min_t
        while current <= max_t:
            slots.append(current)
            current += datetime.timedelta(minutes=intervalo)
        
        # Seleccionar y formatear un slot aleatorio
        return random.choice(slots).strftime(formato)
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


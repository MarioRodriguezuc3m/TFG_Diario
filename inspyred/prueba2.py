from typing import List, Tuple, Dict
from collections import defaultdict
import datetime
import random
import inspyred
from inspyred.swarm import ACS, TrailComponent
import time

class ProblemaAgendamiento:
    def __init__(self, pacientes, consultas, horas, medicos, fases, fases_orden, fases_duracion,args):
        self.pacientes = pacientes
        self.consultas = consultas
        self.horas = horas
        self.medicos = medicos
        self.fases = fases
        self.fases_orden = fases_orden
        self.fases_duracion = fases_duracion
        self.nodos = self.generar_nodos()
        self.args = args
        
    def generar_nodos(self) -> List[TrailComponent]:
        """Crea componentes de rastro con valores iniciales"""
        componentes = []
        for p in self.pacientes:
            for c in self.consultas:
                for h in self.horas:
                    for m in self.medicos:
                        for f in self.fases:
                            duracion = self.fases_duracion[f]
                            valor = 1
                            componentes.append(
                                TrailComponent(
                                    element=(p, c, h, m, f),
                                    value=valor,
                                    maximize=False
                                )
                            )
        return componentes

    def calcular_heuristica(self, node_element: Tuple, current_state: Dict) -> float:
        penalty = 0.0
        paciente, _, hora_str, medico, fase = node_element
        visited = current_state['visited']
        pacientes_progreso = current_state['pacientes_progreso']
        current_paciente = current_state['current_paciente']
        current_fase = current_state['current_fase']

        # 1. Penalizar cambio de paciente si el actual no completó sus fases
        if visited:
            last_element = visited[-1]
            last_paciente = last_element[0]
            if paciente != last_paciente:
                last_fases = pacientes_progreso.get(last_paciente, {})
                if len(last_fases) < len(self.fases_orden):
                    penalty += 1.0

        # 2. Penalizar nuevo paciente que no inicia con fase 1
        if paciente not in pacientes_progreso or not pacientes_progreso[paciente]:
            if self.fases_orden[fase] != 1:
                penalty += 1.0

        # 3. Verificar secuencia de fases para el mismo paciente
        if paciente in pacientes_progreso and pacientes_progreso[paciente]:
            max_orden = max([self.fases_orden[f] for f in pacientes_progreso[paciente]])
            if self.fases_orden[fase] != max_orden + 1:
                penalty += 1.0
            else:
                # Verificar tiempo posterior al final de la fase anterior
                last_fase = [f for f, orden in self.fases_orden.items() if orden == max_orden][0]
                last_end = pacientes_progreso[paciente][last_fase][1]
                current_start = datetime.datetime.strptime(hora_str, "%H:%M").time()
                current_start_min = current_start.hour * 60 + current_start.minute
                if current_start_min < last_end:
                    penalty += 1.0

        # 4. Conflictos de médico en misma hora
        current_start = datetime.datetime.strptime(hora_str, "%H:%M").time()
        current_start_min = current_start.hour * 60 + current_start.minute
        current_end = current_start_min + self.fases_duracion[fase]
        for elem in visited:
            elem_medico = elem[3]
            elem_start = datetime.datetime.strptime(elem[2], "%H:%M").time().hour * 60 + datetime.datetime.strptime(elem[2], "%H:%M").time().minute
            elem_end = elem_start + self.fases_duracion[elem[4]]
            if medico == elem_medico and (current_start_min < elem_end and elem_start < current_end):
                penalty += 1.0

        # 5. Conflictos de fase en misma hora
        for elem in visited:
            if elem[4] == fase:
                elem_start = datetime.datetime.strptime(elem[2], "%H:%M").time().hour * 60 + datetime.datetime.strptime(elem[2], "%H:%M").time().minute
                elem_end = elem_start + self.fases_duracion[elem[4]]
                if (current_start_min < elem_end and elem_start < current_end):
                    penalty += 1.0

        return 1.0 / (1.0 + penalty)

    def construir_solucion(self, random, args, **kwargs) -> List[TrailComponent]:
        solucion = []
        nodos_disponibles = self.nodos.copy()
        visited = []  # Elementos asignados en la solución
        pacientes_progreso = defaultdict(dict)  # {paciente: {fase: (inicio, fin)}}

        for paciente in self.pacientes:
            for fase in sorted(self.fases, key=lambda f: self.fases_orden[f]):
                candidatos = [
                    n for n in nodos_disponibles
                    if n.element[0] == paciente and n.element[4] == fase
                ]
                if not candidatos:
                    break
                # Preparar estado actual para la heurística
                current_state = {
                    'visited': visited.copy(),
                    'pacientes_progreso': pacientes_progreso.copy(),
                    'current_paciente': paciente,
                    'current_fase': fase
                }
                seleccion = self._seleccionar_nodo(random, candidatos, current_state)
                if seleccion is None:
                    break
                solucion.append(seleccion)
                nodos_disponibles.remove(seleccion)
                # Actualizar estado
                elemento = seleccion.element
                visited.append(elemento)
                # Calcular tiempos de la fase actual
                hora_inicio = datetime.datetime.strptime(elemento[2], "%H:%M").time()
                inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
                fin_min = inicio_min + self.fases_duracion[elemento[4]]
                pacientes_progreso[elemento[0]][elemento[4]] = (inicio_min, fin_min)
        return solucion


    def _seleccionar_nodo(self, random, candidatos, current_state):
        if not candidatos:
            return None
        probabilidades = []
        total = 0.0
        for nodo in candidatos:
            feromona = nodo.pheromone
            heuristica = self.calcular_heuristica(nodo.element, current_state)
            prob = (feromona ** self.args['alpha']) * (heuristica ** self.args['beta'])
            probabilidades.append(prob)
            total += prob
        if total == 0:
            return random.choice(candidatos)
        else:
            return random.choices(candidatos, weights=probabilidades, k=1)[0]

    def actualizar_feromonas(self, poblacion):
        """Actualiza feromonas usando las mejores soluciones"""
        for sol in poblacion:
            for componente in sol.candidate:
                tc = next((c for c in self.nodos if c.element == componente), None)
                if tc:
                    tc.pheromone *= (1 - self.args['evaporacion'])
                    tc.pheromone += self.args['deposito'] / (sol.fitness + 1e-6)

    def evaluar(self, candidates,**kwargs) -> List[float]:
        """Calcula el fitness para cada candidato"""
        return [self.calcular_coste(c) for c in candidates]

    def calcular_coste(self, asignaciones: List[Tuple]) -> float:
        """
        Calcula el coste total (penalización) de una asignación de horarios según restricciones:
        1. Solapamientos de médicos/consultas
        2. Orden correcto de fases por paciente
        3. Tiempos entre fases consecutivas
        """
        tiempos_pacientes = defaultdict(list)
        penalty = 0
        fases_activas = []

        # Procesar asignaciones
        for asignacion in asignaciones:
            paciente, consulta, hora_str, medico, fase = asignacion.element
            
            # Conversión de hora a minutos
            hora_inicio = datetime.datetime.strptime(hora_str, "%H:%M").time()
            inicio_min = hora_inicio.hour * 60 + hora_inicio.minute
            fin_min = inicio_min + self.fases_duracion[fase]
            
            # Registrar metadatos
            fases_activas.append({
                'paciente': paciente,
                'consulta': consulta,
                'fase': fase,
                'inicio': inicio_min,
                'fin': fin_min,
                'medico': medico,
                'orden': self.fases_orden[fase]
            })
            tiempos_pacientes[paciente].append((
                self.fases_orden[fase],
                inicio_min,
                fin_min
            ))

        # Verificar solapamientos
        for i in range(len(fases_activas)):
            for j in range(i + 1, len(fases_activas)):
                a = fases_activas[i]
                b = fases_activas[j]
                
                # Conflictos de médico
                if a['medico'] == b['medico'] and (a['inicio'] < b['fin'] and b['inicio'] < a['fin']):
                    penalty += 1000
                    
                # Conflictos de consulta
                if a['consulta'] == b['consulta'] and (a['inicio'] < b['fin'] and b['inicio'] < a['fin']):
                    penalty += 1000

        # Validar secuencia de fases
        for paciente, tiempos in tiempos_pacientes.items():
            tiempos_ordenados = sorted(tiempos, key=lambda x: x[0])  
            orden_esperado = 1
            
            # Orden incorrecto
            for fase_data in tiempos_ordenados:
                if fase_data[0] != orden_esperado:
                    penalty += 2000
                    break
                orden_esperado += 1
            
            # Continuidad temporal
            for i in range(1, len(tiempos_ordenados)):
                prev = tiempos_ordenados[i-1]
                actual = tiempos_ordenados[i]
                if actual[1] < prev[2]:
                    penalty += 2000
                    break
                else:
                    penalty += actual[1] - prev[2]

        return penalty

def ejecutar_optimizacion(config: Dict):
    # Parámetros
    args = {
        'max_generations': 500,
        'alpha': 1.0,
        'beta': 3.0,
        'deposito': 0.5,
        'evaporacion': 0.1,
        'pher': defaultdict(float)
    }
    problema = ProblemaAgendamiento(
        pacientes=config['pacientes'],
        consultas=config['consultas'],
        horas=config['horas'],
        medicos=config['medicos'],
        fases=config['fases'],
        fases_orden=config['fases_orden'],
        fases_duracion=config['fases_duracion'],
        args=args
    )
    # Configurar ACO
    aco = ACS(
        random.Random(),
        components=problema.nodos
    )
    aco.terminator = inspyred.ec.terminators.generation_termination
    inicio = time.time()  # Inicia el cronómetro
    mejor_solucion = aco.evolve(
        generator=problema.construir_solucion,
        evaluator=problema.evaluar,
        pop_size=10, 
        max_generations=args['max_generations'],
        termination_criteria='generations'
    )
    fin= time.time()  # Detiene el cronómetro
    tiempo_ejecucion = fin - inicio
    return mejor_solucion,tiempo_ejecucion

if __name__ == "__main__":
    config = {
        'pacientes': ['P1', 'P2', 'P3','P4'],
        'consultas': ['C1', 'C2','C3'],
        'horas': ['09:00', '10:00', '11:00', '12:00', '13:00','14:00','15:00','16:00','17:00','18:00'],
        'medicos': ['M1', 'M2', 'M3'],
        'fases': ['F1', 'F2', 'F3'],
        'fases_orden': {'F1': 1, 'F2': 2, 'F3': 3},
        'fases_duracion': {'F1': 60, 'F2': 60, 'F3': 60}
    }

    resultado,tiempo_ejecucion = ejecutar_optimizacion(config)
    # Suponiendo que resultado es una lista de soluciones, accedemos al mejor candidato:
    mejor_solucion = min(resultado, key=lambda sol: sol.fitness)

    # Ahora puedes acceder a su fitness:
    print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")
    print("\n● Mejor Horario ●")
    print(f"Coste Total: {mejor_solucion.fitness}")
    for asignacion in mejor_solucion.candidate:
        # Suponiendo que 'asignacion' es un objeto TrailComponent:
        print(f"· {asignacion.element[0]} | {asignacion.element[1]} | {asignacion.element[2]} | {asignacion.element[3]} | {asignacion.element[4]}")


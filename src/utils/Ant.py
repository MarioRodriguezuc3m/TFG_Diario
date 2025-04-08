import random
from utils.generate_graph_components import *
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import datetime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Standard.Graph import Graph  # Se usa solo para type hint

class Ant:
    def __init__(self, graph: "Graph", fases_orden: Dict[str, int], fases_duration: Dict[str, int], pacientes: List[str], alpha: float = 1.0, beta: float = 1.0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.visited: List[Tuple] = []
        self.fases_orden = fases_orden  
        self.fases_duration = fases_duration  
        self.pacientes = pacientes
        self.pacientes_progreso = defaultdict(dict, {paciente: {} for paciente in self.pacientes}) #Diccionario para mantener el progreso de cada paciente
        self.current_node: Tuple = None
        self.total_cost: float = 0.0
        self.valid_solution = False

    def choose_next_node(self) -> Tuple:
        if self.current_node is None:
            # Elegir aleatoriamente el primer nodo (solo nodos iniciales válidos)
            valid_initial_nodes = [
                node for node in self.graph.nodes
                if self.fases_orden[node[4]] == 1  # Solo Fase1 (orden 1)
            ]
            return random.choice(valid_initial_nodes) if valid_initial_nodes else None
        else:
            candidates = self.graph.edges.get(self.current_node, [])
            if not candidates:
                return None

            candidate_list = []
            probabilities = []
            total = 0.0

            for node in candidates:
                heuristic = self.calcular_heuristica(node)
                # Obtener el valor de la feromona para la transición actual
                pheromone = self.graph.get_pheromone(self.current_node, node)
                candidate_weight = (pheromone ** self.alpha) * (heuristic ** self.beta)
                candidate_list.append(node)
                probabilities.append(candidate_weight)
                total += candidate_weight

            if total == 0:
                return random.choice(candidate_list)

            return random.choices(candidate_list, weights=[p / total for p in probabilities], k=1)[0]

    def calcular_heuristica(self, node: Tuple) -> float:
        """
        Calcula la heurística para un nodo candidato con un enfoque más sofisticado
        que considera múltiples factores con pesos diferenciados.
        """
        paciente, consulta, hora, medico, fase = node
        
        # Pesos para los diferentes factores
        W_CONFLICTO = 4.0    # Importancia de evitar conflictos
        W_TIEMPO = 2.0       # Importancia de la distribución temporal
        
        # Inicializar puntuación base
        score = 10.0  # Empezamos con valor positivo y aplicamos modificadores
        
        # 1. FACTOR DE TIEMPO: Verificar distribución temporal adecuada
        if self.current_node:
            # Verificar tiempo adecuado entre fases consecutivas del mismo paciente
            if paciente == self.current_node[0]:
                # Conversión de horas a minutos para comparación
                hora_actual = datetime.datetime.strptime(self.current_node[2], "%H:%M")
                hora_siguiente = datetime.datetime.strptime(hora, "%H:%M")
                
                # Minutos desde medianoche
                min_actual = hora_actual.hour * 60 + hora_actual.minute
                min_siguiente = hora_siguiente.hour * 60 + hora_siguiente.minute
                
                # Duración de la fase actual
                duracion_actual = self.fases_duration[self.current_node[4]]
                
                # Verificar que la fase siguiente comience después de que termine la actual
                if min_siguiente < min_actual + duracion_actual:
                    score -= 10.0 * W_TIEMPO  # Penalización fuerte por solapamiento temporal
                else:
                    # Tiempo de espera entre fin de fase actual e inicio de la siguiente
                    tiempo_espera = min_siguiente - (min_actual + duracion_actual)
                    
                    # Evaluar tiempo de espera
                    if tiempo_espera <= 30:
                        score += 5.0 * W_TIEMPO  # Bonificación por continuidad temporal óptima
                    elif tiempo_espera <= 60:
                        score += 3.0 * W_TIEMPO  # Bonificación por tiempo de espera razonable
                    elif tiempo_espera > 180:  # Más de 3 horas
                        score -= (tiempo_espera - 180) / 60.0 * W_TIEMPO  # Penalización gradual por espera excesiva
        
        # 2. FACTOR DE CONFLICTOS: Verificar conflictos de recursos
        conflicto_medico = False
        conflicto_consulta = False
        
        for nodo_visitado in self.visited:
            v_paciente, v_consulta, v_hora, v_medico, v_fase = nodo_visitado
            
            # Convertir horas a minutos para comparación
            hora_visitada = datetime.datetime.strptime(v_hora, "%H:%M")
            hora_candidata = datetime.datetime.strptime(hora, "%H:%M")
            
            # Calcular rangos de tiempo
            v_inicio = hora_visitada.hour * 60 + hora_visitada.minute
            v_fin = v_inicio + self.fases_duration[v_fase]
            c_inicio = hora_candidata.hour * 60 + hora_candidata.minute
            c_fin = c_inicio + self.fases_duration[fase]
            
            # Verificar solapamiento temporal
            if c_inicio < v_fin and v_inicio < c_fin:
                # Penalizar conflictos de recursos
                if v_medico == medico:
                    conflicto_medico = True
                if v_consulta == consulta:
                    conflicto_consulta = True
        
        # Aplicar penalizaciones por conflictos
        if conflicto_medico:
            score -= 15.0 * W_CONFLICTO  # Penalización severa por conflicto de médico
        if conflicto_consulta:
            score -= 15.0 * W_CONFLICTO  # Penalización severa por conflicto de consulta
        
        # 3. FACTOR DE BALANCEO: Promover distribución equilibrada
        # Contar cuántas veces ha aparecido cada médico y consulta
        conteo_medicos = {}
        conteo_consultas = {}
        
        for nodo in self.visited:
            conteo_medicos[nodo[3]] = conteo_medicos.get(nodo[3], 0) + 1
            conteo_consultas[nodo[1]] = conteo_consultas.get(nodo[1], 0) + 1
        
        # Bonificar recursos menos utilizados
        medico_count = conteo_medicos.get(medico, 0)
        consulta_count = conteo_consultas.get(consulta, 0)
        
        # Normalizar por el promedio
        avg_medicos = sum(conteo_medicos.values()) / max(1, len(conteo_medicos))
        avg_consultas = sum(conteo_consultas.values()) / max(1, len(conteo_consultas))
        
        if medico_count < avg_medicos:
            score += 1.0  # Ligera bonificación por usar médicos menos cargados
        if consulta_count < avg_consultas:
            score += 1.0  # Ligera bonificación por usar consultas menos cargadas
        
        # Asegurar que la heurística sea positiva (evitar división por cero o valores negativos)
        return max(0.1, score)

    def move(self, node: Tuple):
        self.current_node = node
        self.visited.append(node)
        paciente, _, hora, _ , fase = node = node
        self.pacientes_progreso[paciente][fase] = hora
        
        # Verificar solución completa y orden correcto
        self.valid_solution = all(
            self._fases_en_orden_correcto(fases) 
            for fases in self.pacientes_progreso.values()
        )

    def _fases_en_orden_correcto(self, fases_paciente: Dict) -> bool:
        """Verifica si las fases de un paciente están en el orden correcto."""
        fases = list(fases_paciente.keys())
        orden = [self.fases_orden[f] for f in fases]
        return orden == sorted(orden) and len(orden) == len(self.fases_orden)
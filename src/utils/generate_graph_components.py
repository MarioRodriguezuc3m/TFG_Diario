from typing import List, Tuple, Dict
from collections import defaultdict

def generar_nodos(pacientes, consultas, horas, medicos, fases) -> List[Tuple]:
    """
    Genera todos los nodos posibles del grafo combinando cada parámetro.
    
    Cada nodo representa una posible asignación de:
    (paciente, consulta, hora, médico, fase)
    """
    return [(p, c, h, m, f) for p in pacientes for c in consultas for h in horas for m in medicos for f in fases]

def generar_aristas(nodos: List[Tuple], orden_fases: Dict) -> Dict[Tuple, List[Tuple]]:
    """
    Genera las aristas del grafo aplicando restricciones de secuenciación.
    
    Las aristas conectan nodos válidos según:
    - Restricciones de tiempo (médico/consulta no repetidos en misma hora)
    - Secuencia de fases para el mismo paciente
    - Transición entre último fase de un paciente y primer fase de otro
    """
    aristas = defaultdict(list)
    max_orden_fases = max(orden_fases.values())  # Determina la última fase

    # Generar todas las posibles conexiones entre nodos
    for nodo1 in nodos:
        for nodo2 in nodos:
            if nodo1 == nodo2:
                continue  # Evitar conexiones consigo mismo

            # Descomponer nodos en sus componentes
            p1, c1, h1, m1, f1 = nodo1
            p2, c2, h2, m2, f2 = nodo2

            # Restricción: No puede haber dos asignaciones con el mismo médico o misma consulta en la misma hora
            if (m1 == m2 and h1 == h2) or (c1 == c2 and h1 == h2):
                continue  # Conexión inválida

            # Para el mismo paciente - Conectar únicamente fases consecutivas
            if p1 == p2:
                # Solo se conecta si la fase destino es la siguiente en la secuencia
                if orden_fases[f2] == orden_fases[f1] + 1:
                    aristas[nodo1].append(nodo2)
            
            # Diferentes pacientes - Conectar fin->inicio
            else:
                # Solo se conecta si la última fase de paciente1 con primera fase de paciente2
                if orden_fases[f1] == max_orden_fases and orden_fases[f2] == 1:
                    aristas[nodo1].append(nodo2)

    return aristas  # Corregido: se eliminó texto adicional erróneo
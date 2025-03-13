from typing import List, Tuple, Dict
from collections import defaultdict

def generar_nodos(pacientes, consultas, horas, medicos, fases) -> List[Tuple]:
    return [(p, c, h, m, f) for p in pacientes for c in consultas for h in horas for m in medicos for f in fases]

def generar_aristas(nodos: List[Tuple], orden_fases: Dict) -> Dict[Tuple, List[Tuple]]:
    aristas = defaultdict(list)
    # Determinar el máximo orden (última fase) a partir del diccionario de fases
    max_orden_fases = max(orden_fases.values())
    
    for nodo1 in nodos:
        for nodo2 in nodos:
            if nodo1 == nodo2:
                continue  # No conectar un nodo consigo mismo
            
            p1, c1, h1, m1, f1 = nodo1
            p2, c2, h2, m2, f2 = nodo2
            
            # Restricciones de hora:
            # Evitar asignar el mismo médico a la misma hora o la misma consulta a la misma hora.
            if (m1 == m2 and h1 == h2) or (c1 == c2 and h1 == h2):
                continue
            
            if p1 == p2:
                # Conexión dentro del mismo paciente:
                # Conectar solo si la fase destino es la siguiente en el orden respecto a la fase origen.
                if orden_fases[f2] == orden_fases[f1] + 1:
                    aristas[nodo1].append(nodo2)
            else:
                # Conexión entre diferentes pacientes:
                # Conectar únicamente la última fase de un paciente (nodo1) con la primera fase de otro (nodo2).
                if orden_fases[f1] == max_orden_fases and orden_fases[f2] == 1:
                    aristas[nodo1].append(nodo2)
                    
    return aristas
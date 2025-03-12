from typing import List, Dict, Tuple
from collections import defaultdict

# =============================================
# Funciones para generar nodos y aristas
# =============================================
def generar_nodos(pacientes, consultas, horas, medicos) -> List[Tuple]:
    return [(p, c, h, m) for p in pacientes for c in consultas for h in horas for m in medicos]

def generar_aristas(nodos: List[Tuple], orden_consultas: Dict) -> Dict[Tuple, List[Tuple]]:
    aristas = defaultdict(list)
    # Determinar el máximo orden (última consulta) a partir del diccionario
    max_orden = max(orden_consultas.values())
    
    for nodo1 in nodos:
        for nodo2 in nodos:
            if nodo1 == nodo2:
                continue  # No conectar un nodo consigo mismo
                
            p1, c1, h1, m1 = nodo1
            p2, c2, h2, m2 = nodo2
            
            # Restricciones de hora:
            # Evitar asignar el mismo médico a la misma hora o la misma consulta a la misma hora.
            if (m1 == m2 and h1 == h2) or (c1 == c2 and h1 == h2):
                continue
            
            if p1 == p2:
                # Conexión dentro del mismo paciente:
                # Conectar solo si la consulta destino es la siguiente en el orden respecto a la consulta origen.
                if orden_consultas[c2] == orden_consultas[c1] + 1:
                    aristas[nodo1].append(nodo2)
            else:
                # Conexión entre diferentes pacientes:
                # Conectar únicamente la última consulta de un paciente (nodo1) con la primera consulta de otro (nodo2).
                if orden_consultas[c1] == max_orden and orden_consultas[c2] == 1:
                    aristas[nodo1].append(nodo2)
                    
    return aristas
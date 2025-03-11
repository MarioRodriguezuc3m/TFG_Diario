# Supongamos que tenemos las siguientes listas de entrada:
pacientes = ['Paciente1', 'Paciente2']
consultas = ['ConsultaA', 'ConsultaB']  # Se asume que deben seguir un orden (p.ej., ConsultaA antes que ConsultaB)
horas = ['09:00', '10:00']
medicos = ['MedicoX', 'MedicoY']

# Para el criterio de orden, definimos un orden de consultas:
orden_consultas = {'ConsultaA': 1, 'ConsultaB': 2}

import random
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# =============================================
# Funciones para generar nodos y aristas
# =============================================
def generar_nodos(pacientes, consultas, horas, medicos) -> List[Tuple]:
    return [(p, c, h, m) for p in pacientes for c in consultas for h in horas for m in medicos]

def generar_aristas(nodos: List[Tuple], orden_consultas: Dict) -> Dict[Tuple, List[Tuple]]:
    aristas = defaultdict(list)
    for nodo1 in nodos:
        for nodo2 in nodos:
            if nodo1 == nodo2:
                continue
                
            p1, c1, h1, m1 = nodo1
            p2, c2, h2, m2 = nodo2
            
            # Restricciones
            misma_hora_medico = (m1 == m2 and h1 == h2)
            misma_hora_consulta = (c1 == c2 and h1 == h2)
            
            if not misma_hora_medico and not misma_hora_consulta:
                aristas[nodo1].append(nodo2)
    
    return aristas
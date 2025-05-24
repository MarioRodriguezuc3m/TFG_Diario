from typing import List, Tuple, Dict, Any
from collections import defaultdict

def construir_mapeo_paciente_info(tipos_estudio_data: List[Dict]) -> Dict:
    """
    Crea un mapeo de cada paciente a la información relevante de su estudio,
    incluyendo el orden de sus fases y la fase máxima.
    """
    paciente_a_estudio_info = {}
    for estudio in tipos_estudio_data:
        nombre_estudio = estudio["nombre_estudio"]
        orden_fases_estudio = estudio["orden_fases"]
        max_orden_estudio = 0
        if orden_fases_estudio: # Asegurarse de que orden_fases_estudio no esté vacío
            max_orden_estudio = max(orden_fases_estudio.values())

        for paciente in estudio["pacientes"]:
            paciente_a_estudio_info[paciente] = {
                "orden_fases": orden_fases_estudio,
                "max_orden": max_orden_estudio,
                "nombre_estudio": nombre_estudio
            }
    return paciente_a_estudio_info

def generar_nodos(config_data: Dict[str, Any], horas_disponibles: List[str]) -> List[Tuple]:
    """
    Genera todos los nodos posibles del grafo combinando cada parámetro
    para múltiples tipos de estudio. La información de pacientes, fases,
    consultas, y médicos se extrae de config_data. Las horas disponibles
    son pasadas como argumento.
    
    Cada nodo representa una posible asignación de:
    (paciente, consulta, hora, médico, fase)
    """
    nodos = []
    consultas = config_data["consultas"]
    medicos = config_data["medicos"]

    for estudio in config_data["tipos_estudio"]:
        pacientes_estudio = estudio["pacientes"]
        fases_estudio = estudio["fases"] 
        for p in pacientes_estudio:
            for c in consultas:
                for h in horas_disponibles:
                    for m in medicos:
                        for f in fases_estudio:
                            nodos.append((p, c, h, m, f))
    return nodos

def generar_aristas(nodos: List[Tuple], paciente_info: Dict[str, Dict[str, Any]]) -> Dict[Tuple, List[Tuple]]:
    """
    Genera las aristas del grafo aplicando restricciones de secuenciación
    para múltiples tipos de estudio. Utiliza paciente_info para obtener
    el orden de fases específico de cada paciente.
    
    Las aristas conectan nodos válidos según:
    - Restricciones de tiempo (médico/consulta no repetidos en misma hora)
    - Secuencia de fases para el mismo paciente (según su tipo de estudio)
    - Transición entre última fase de un paciente y primera fase de otro (considerando sus tipos de estudio)
    """
    aristas = defaultdict(list)

    for nodo1 in nodos:
        for nodo2 in nodos:
            if nodo1 == nodo2:
                continue

            p1, c1, h1, m1, f1 = nodo1
            p2, c2, h2, m2, f2 = nodo2

            # Restricción de recursos: mismo médico o consulta no pueden estar ocupados a la misma hora
            if h1 == h2 and (m1 == m2 or c1 == c2):
                continue

            # Obtener información de los estudios de cada paciente
            info_p1 = paciente_info.get(p1)
            info_p2 = paciente_info.get(p2)

            if not info_p1 or not info_p2:
                raise ValueError(f"Información de paciente no encontrada para {p1} o {p2}")
            
            # Verificar que las fases pertenezcan a los estudios correspondientes
            if f1 not in info_p1["orden_fases"] or f2 not in info_p2["orden_fases"]:
                raise ValueError(f"Fase {f1} o {f2} no pertenece al estudio del paciente {p1} o {p2}")

            orden_fases_p1 = info_p1["orden_fases"]
            max_orden_p1 = info_p1["max_orden"]
            orden_fases_p2 = info_p2["orden_fases"]

            # Caso 1: Mismo paciente - debe seguir secuencia de fases
            if p1 == p2:
                # f2 debe ser la siguiente fase después de f1
                if orden_fases_p1.get(f2, -999) == orden_fases_p1.get(f1, -998) + 1:
                    aristas[nodo1].append(nodo2)
            # Caso 2: Diferentes pacientes - transición de última fase de p1 a primera fase de p2
            else:
                # f1 debe ser la última fase de p1 y f2 debe ser la primera fase de p2
                if orden_fases_p1.get(f1, -999) == max_orden_p1 and orden_fases_p2.get(f2, -999) == 1:
                    aristas[nodo1].append(nodo2)
                    
    return aristas
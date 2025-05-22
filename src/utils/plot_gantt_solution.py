import plotly.figure_factory as ff
import pandas as pd
from datetime import datetime, timedelta
import os

def plot_gantt_chart(best_solution, fases_duration, pacientes, medicos, consultas, save_path='/app/plots/'):
    """
    Creates a Gantt chart from the ACO solution using Plotly and saves it as PNG to the specified path.
    
    Args:
        best_solution: List of assignments as tuples (patient, consultation, time, doctor, phase)
        fases_duration: Dictionary mapping phases to their duration in minutes
        pacientes: List of patient names
        medicos: List of doctor names
        consultas: List of consultation rooms
        save_path: Path to save the Gantt chart
    """
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Parse the solution into a structured format for Plotly
    gantt_data = []
    
    # Base date for plotting (just using today's date since we only care about time)
    base_date = datetime.today().date()
    
    # Define colors for different phases
    phase_colors = {
        'Fase1': 'rgb(52, 152, 219)',   # Azul claro (Peter River)
        'Fase2': 'rgb(231, 76, 60)',    # Rojo (Alizarin)
        'Fase3': 'rgb(46, 204, 113)',   # Verde (Emerald)
        'Fase4': 'rgb(155, 89, 182)',   # Púrpura (Amethyst)
    }

    
    # Definir hora de inicio y fin para el eje X
    start_hour = 9  # Hora de inicio: 9:00
    end_hour = 19   # Hora de fin: 19:00 
    
    for assignment in best_solution:
        patient, consultation, start_time_str, doctor, phase = assignment
        
        try:
            # Convert time string to datetime object
            hour, minute = map(int, start_time_str.split(':'))
            start_time = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
            duration = fases_duration.get(phase, 60)  # Default 60 minutes if not found
            end_time = start_time + timedelta(minutes=duration)
            
            # Format dates for Plotly
            start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get color for this phase
            color = phase_colors.get(phase, 'rgb(128, 128, 128)')  # Default gray if phase not found
            
            gantt_data.append({
                'Task': f"{patient}",
                'Start': start_str,
                'Finish': end_str,
                'Resource': phase,
                'Description': f"Patient: {patient}<br>Doctor: {doctor}<br>Phase: {phase}<br>Room: {consultation}<br>Time: {start_time_str} - {end_time.strftime('%H:%M')}",
                'Room': consultation
            })
            
            gantt_data.append({
                'Task': f"{doctor}",
                'Start': start_str,
                'Finish': end_str,
                'Resource': phase,
                'Description': f"Patient: {patient}<br>Doctor: {doctor}<br>Phase: {phase}<br>Room: {consultation}<br>Time: {start_time_str} - {end_time.strftime('%H:%M')}",
                'Room': consultation
            })
            gantt_data.append({
                'Task': f"{consultation}",
                'Start': start_str,
                'Finish': end_str,
                'Resource': phase,
                'Description': f"Patient: {patient}<br>Doctor: {doctor}<br>Phase: {phase}<br>Room: {consultation}<br>Time: {start_time_str} - {end_time.strftime('%H:%M')}",
                'Room': consultation
            })
        except ValueError as e:
            print(f"Error parsing time from {start_time_str}: {e}")
    
    if not gantt_data:
        print("No valid schedule data found. Check your solution format.")
        return None
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(gantt_data)
    df = df.sort_values(by=['Task', 'Start'])
    
    # Create the Gantt chart
    fig = ff.create_gantt(
        df,
        colors={phase: phase_colors.get(phase, 'rgb(128, 128, 128)') for phase in df['Resource'].unique()},
        index_col='Resource',
        show_colorbar=True,
        group_tasks=True,
        showgrid_x=True,
        showgrid_y=True,
        title='Medical Appointments Schedule',
        height=600,
        width=1000,
        bar_width=0.4,
        show_hover_fill=True
    )
    # Establece la opacidad de todas las barras a 0.5
    for shape in fig.data:
        shape['opacity'] = 0.5
    
    # Crear marcas de tiempo para cada hora desde las 9:00 hasta las 19:00
    time_ticks = []
    time_labels = []
    for hour in range(start_hour, end_hour + 1):
        tick_time = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hour)
        time_ticks.append(tick_time.strftime('%Y-%m-%d %H:%M:%S'))
        time_labels.append(f"{hour}:00")
    
    # Ordenar la leyenda en orden específico
    ordered_phases = ['Fase1', 'Fase2', 'Fase3', 'Fase4']
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Horas",
        yaxis_title="Recursos",
        legend_title="Fases",
        font=dict(size=12),
        xaxis=dict(
            tickvals=time_ticks,  # Valores para las marcas de tiempo
            ticktext=time_labels,  # Etiquetas de texto para las marcas
            tickmode='array',     # Modo array para usar valores personalizados
            range=[                # Rango del eje X
                datetime.combine(base_date, datetime.min.time()) + timedelta(hours=start_hour),
                datetime.combine(base_date, datetime.min.time()) + timedelta(hours=end_hour)
            ]
        ),
        updatemenus=[]  # Elimina los botones de selección de período
    )
    
    # Reordenar la leyenda
    for i, p in enumerate(fig.data):
        if p.name in ordered_phases:
            p.legendgroup = str(ordered_phases.index(p.name))
            p.legendrank = ordered_phases.index(p.name)
    
    # Save the figure as PNG only
    file_path = os.path.join(save_path, 'schedule_gantt.png')
    fig.write_image(file_path, scale=2)  # scale=2 for higher resolution
    
    print(f"Gantt chart saved to {file_path}")
    return file_path
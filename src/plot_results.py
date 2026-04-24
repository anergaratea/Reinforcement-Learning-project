import os
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter

def plot_learning_curve(log_folder, title="Curva de Aprendizaje"):
    """
    Lee los resultados del Monitor de Stable-Baselines3 y genera una gráfica
    de la recompensa obtenida por episodio durante el entrenamiento.
    """
    print(f"Cargando logs desde: {log_folder}")
    
    # results_plotter.plot_results toma la carpeta, timesteps, el eje X y el título
    try:
        results_plotter.plot_results(
            [log_folder], 
            1e5, # número máximo de timesteps estimado
            results_plotter.X_TIMESTEPS, 
            title
        )
        plt.show()
    except Exception as e:
        print(f"No se pudieron graficar los resultados. ¿Aseguraste ejecutar train.py primero? Error: {e}")

if __name__ == "__main__":
    log_folder = "logs"
    plot_learning_curve(log_folder, title="PPO en CartPole-v1")

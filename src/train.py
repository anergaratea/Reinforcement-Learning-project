import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os

def main():
    # Configuración de directorios
    env_id = "CartPole-v1"
    models_dir = "models/PPO"
    logdir = "logs"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Crear el entorno
    env = gym.make(env_id)
    # Monitorizar el entorno y guardar los logs (CSV) en la carpeta de logs
    env = Monitor(env, logdir)

    # Inicializar el modelo PPO (Proximal Policy Optimization)
    # MlpPolicy usa una red neuronal estándar (Multi-Layer Perceptron) adecuada para CartPole
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    # Entrenar el agente
    TIMESTEPS = 20000
    print(f"--- Empezando entrenamiento con {TIMESTEPS} timesteps en el entorno {env_id} ---")
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO_CartPole")

    # Guardar el modelo
    model_path = f"{models_dir}/{env_id}_model"
    model.save(model_path)
    print(f"--- Modelo guardado en {model_path}.zip ---")

    # Evaluar el modelo entrenado
    print("--- Evaluando el modelo ---")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Recompensa media sobre 10 episodios: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()

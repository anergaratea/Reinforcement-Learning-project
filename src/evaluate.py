import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

def main():
    env_id = "CartPole-v1"
    model_path = f"models/PPO/{env_id}_model.zip"

    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}.")
        print("Asegúrate de ejecutar primero: python src/train.py")
        return

    print(f"Cargando modelo desde {model_path}...")
    model = PPO.load(model_path)

    # Crear el entorno con render_mode="human" para poder ver el agente jugando
    env = gym.make(env_id, render_mode="human")

    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        print(f"Iniciando Episodio {ep + 1}...")
        while not done and not truncated:
            # Predecir la mejor acción según el modelo entrenado
            action, _states = model.predict(obs, deterministic=True)
            
            # Ejecutar la acción en el entorno
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Pausa breve para que la animación sea visible a velocidad humana
            time.sleep(0.02)
            
        print(f"Episodio {ep + 1} terminado. Recompensa total = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()

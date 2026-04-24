# Reinforcement Learning Project con Stable-Baselines3

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-brightgreen.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-blueviolet.svg)

Bienvenido a un proyecto inicial de Aprendizaje por Refuerzo (RL) usando [Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/).
SB3 es una biblioteca que proporciona implementaciones fiables, modernas y optimizadas de algoritmos de RL basados en PyTorch (PPO, A2C, SAC, DQN, TD3, etc.).

## Estructura del Proyecto

* `src/train.py`: Carga el entorno `CartPole-v1`, inicializa un agente basado en el algoritmo **PPO**, lo entrena y guarda el modelo resultante. Genera logs en CSV y formato TensorBoard.
* `src/evaluate.py`: Carga el agente preentrenado y ejecuta el entorno en modo visual (`render_mode="human"`) para que puedas observar lo bien que ha aprendido la tarea.
* `src/plot_results.py`: Script con Matplotlib para generar rápidamente la curva de aprendizaje a partir de los datos registrados.
* `requirements.txt`: Dependencias principales del proyecto.

## Instalación

1. Clona el repositorio e ingresa en él.
2. (Recomendado) Crea un entorno virtual (`venv` o `conda`).
3. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Uso Básico

### 1. Entrenar el Agente

Abre tu terminal y ejecuta:

```bash
python src/train.py
```

Durante el entrenamiento verás mensajes de logging desde SB3 que te dirán cómo evoluciona la recompensa. Al terminar, tu modelo se guardará en la carpeta `models/PPO/`.

### 2. Ver al Agente en Acción

Una vez completado el entrenamiento, puedes ver el resultado de la política visualmente:

```bash
python src/evaluate.py
```

Esto abrirá una pequeña animación de "CartPole" (el carrito intentando balancear el palo vertical).

### 3. Visualizar y Analizar el Aprendizaje

La visualización de métricas te permite entender cómo mejorar al agente:

**Opción A: Gráfico Rápido (Matplotlib)**
Ejecuta el script creado para observar la evolución de la recompensa por episodio:
```bash
python src/plot_results.py
```

**Opción B: Panel Interactivo (TensorBoard)**
Monitoriza métricas avanzadas en tiempo real (pérdidas, duración de episodio, etc):
```bash
tensorboard --logdir=logs
```
Luego, entra en `http://localhost:6006` en tu navegador.

## Resultados Esperados

Al entrenar el agente con el algoritmo **PPO** en el entorno `CartPole-v1` por unos `20,000` timesteps, observarás los siguientes resultados:

- **Convergencia rápida:** La recompensa máxima en este entorno suele ser de 500 por episodio. El agente típicamente aprende a balancear el palo perfectamente antes de terminar el entrenamiento, logrando recompensas de `500` (o muy cercanas) de manera consistente.
- **Evaluación estable:** Durante la ejecución de `src/evaluate.py`, notarás que el poste rara vez se cae. El carrito simplemente hace micro-ajustes de izquierda a derecha.
- **Curva de Aprendizaje:** Al correr TensorBoard o `src/plot_results.py`, la gráfica de recompensa media subirá abruptamente en los primeros miles de pasos y luego se estabilizará en la recompensa máxima permitida, demostrando que PPO es muy eficiente resolviendo entornos de dimensión reducida.

## Ideas de Extensión
- **Entornos Personalizados:** Entrenar un agente en tu propio entorno personalizado heredando de la clase padre `gymnasium.Env`.
- **Nuevos Algoritmos:** Probar **DQN** en entornos clásicos o de Atari (requiere la variante extendida de gimnasio).
- **MLOps:** Integrar [W&B (Weights & Biases)](https://wandb.ai/) para tracking centralizado del ecosistema si pretendes escalar el entrenamiento de este proyecto.

## Referencias
- [Documentación Oficial de Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Documentación de Gymnasium (Farama Foundation)](https://gymnasium.farama.org/)
- [Curso de Aprendizaje por Refuerzo de Hugging Face](https://huggingface.co/learn/deep-rl-course/): Excelente recurso para dominar las bases que utiliza SB3.
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/en/latest/): Fundamentos teóricos exhaustivos de algoritmos actor-crítico como PPO y SAC.

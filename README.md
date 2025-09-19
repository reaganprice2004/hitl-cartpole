# Human-in-the-Loop (HITL) Control System — CartPole DQN Example

> **Brief overview:** This repository contains a compact, well-documented Human-in-the-Loop (HITL) example that trains a DQN agent to solve **CartPole-v1** while ingesting *human corrections / demonstrations* captured in a separate teleoperation session. The goal is to demonstrate how human feedback can be integrated into RL training pipelines for safer, more sample-efficient learning — a relevant skill in Intelligent Systems Engineering (aerospace, healthcare, autonomy).

---

## Table of Contents

- [Why this project?](#why-this-project)  
- [What you'll find here](#what-youll-find-here)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [1) Record human corrections](#1-record-human-corrections)  
  - [2) Train the agent (with human corrections)](#2-train-the-agent-with-human-corrections)  
  - [3) Run the trained agent](#3-run-the-trained-agent)  
- [Examples & Use cases](#examples--use-cases)  
- [Design notes & implementation details](#design-notes--implementation-details)  
- [Extending this project](#extending-this-project)  
- [License](#license)

---

## Why this project?

Recruiters in ISE look for candidates who can combine control systems, machine learning, and practical engineering trade-offs. This repo demonstrates:
- Integration of **human demonstrations/corrections** into an RL training loop.  
- Clear, modular code with extensive inline comments for readability.  
- A reproducible pipeline you can expand to more complex control problems (quadrotors, manipulators, autonomous vehicles).

---

## What you'll find here

- `agent.py` — Minimal DQN agent with per-line comments.  
- `replay_buffer.py` — Simple FIFO replay buffer.  
- `human_recorder.py` — Run this CLI script to tele-operate the environment and log human corrections to `human_corrections.csv`.  
- `train.py` — Training loop that periodically ingests human corrections and prioritizes their loss.  
- `models.py` — Optional model definitions.  
- `README.md` — This document.  
- `LICENSE` — MIT license.

---

## Installation

> Tested on Ubuntu / macOS. For Windows, the keyboard capture in `human_recorder.py` may need adjustments.

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/hitl-cartpole.git
cd hitl-cartpole
```

2. (Optional) Create Python virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows (PowerShell)
```

3. Install dependencies (minimal):
```bash
pip install torch numpy gymnasium
```

4. (Optional: for visualization) Install gym[box2d] or other render extras if you use a different env.

---

## Usage

### 1) Record human corrections

Open a terminal and run the human teleoperation recorder. This script will render the environment and wait for keypresses:
- `a` -> push cart left (action 0)  
- `d` -> push cart right (action 1)  
- `q` -> quit and save

```bash
python3 human_recorder.py --env CartPole-v1 --out human_corrections.csv
```

Each time you tele-operate, transitions are appended to `human_corrections.csv`. Keep this running in a separate terminal/session when you want to capture demonstrations or corrections while watching the agent run.

> Note: `human_recorder.py` uses a Unix `getch()` implementation for simplicity. On Windows you may need to replace it with `msvcrt.getch()` or use a cross-platform library like `pynput` or `pygame`.

### 2) Train the agent (with human corrections)

Run the training script. The training loop will periodically read `human_corrections.csv` and add those transitions into the replay buffer with `is_human=True`. The agent's loss scales human examples more heavily to prioritize learning from them.

```bash
python3 train.py
```

Important flags & knobs are in `train.py` (episodes, ingest interval, human priority). You can tune these to test how much human data impacts learning speed and stability.

### 3) Run the trained agent

After training, a model is saved to `dqn_agent_final.pth`. You can write a small `run_agent.py` to load the model and run the environment while rendering.

```python
# minimal example (create run_agent.py)
import gymnasium as gym, torch
from agent import DQNAgent
env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
agent.q_net.load_state_dict(torch.load("dqn_agent_final.pth"))
state, info = env.reset()
done = False
while not done:
    env.render()
    action = agent.select_action(state, epsilon=0.0)  # greedy
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

---

## Examples & Use cases

- **Aerospace:** Human pilots provide corrective commands during early autonomous flight trials (pilot-in-loop). Use recorded pilot corrections to bootstrap an autopilot.  
- **Healthcare:** Clinicians supply occasional corrections to an assistive robot; those corrections are prioritized in learning to ensure patient safety.  
- **Autonomy research:** Test how varying the fraction of human corrections affects sample efficiency and safety.

Include GIFs or demo videos in the README for visual impact. You can record screen captures while running `human_recorder.py` and `train.py` and add them to the repo (e.g., `media/teleop.gif`).

---

## Design notes & implementation details

- `human_recorder.py` saves transitions as CSV rows: (`state`, `action`, `next_state`, `reward`, `done`). The training loop converts these into transitions with `is_human=True`.  
- We prioritize human examples by scaling their loss (simple approach). More advanced methods: importance sampling, prioritized replay, inverse RL, or using human data to pretrain the policy.  
- Safety tip: always inspect your human corrections and remove noisy/outlier entries. The current pipeline appends; you might want a deduplication/validation step.

---

## Extending this project

- Replace DQN with **PPO** or **SAC** for better continuous-control performance.  
- Swap CartPole for a simulated quadrotor environment (AirSim / PyBullet / MuJoCo).  
- Add a lightweight web dashboard (Streamlit / Flask) to visualize agent vs human trajectories and allow label corrections in-browser.  
- Implement prioritized replay that uses TD-error + human flag for more principled prioritization.

---

## License

This project is released under the **MIT License** — see `LICENSE` for details.

---

> If you'd like, I can also scaffold the `run_agent.py` file, add helper scripts for visualization, or generate an example GIF demonstrating training vs. human corrections.
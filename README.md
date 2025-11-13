# 🧠 Society Simulation — Interactive AI World

An **interactive, learning-based simulation of a miniature society**, where multiple AI agents coexist, compete, and cooperate inside a shared environment.  
Each agent is driven by an independent **Actor–Critic reinforcement learning model** that adapts behavior through reward signals — creating emergent social patterns like cooperation, trade, and conflict.

This project runs best inside **Jupyter notebooks**, where you can visualize both the evolving **social relationships** and **geographical world** in real time, then replay the simulation step by step.

---

## 🚀 Features

### 🌍 World Engine
- 2D **cellular grid** representing the environment.
- Each cell contains a renewable resource (`stock`) that regenerates at a configurable rate.
- Agents can move between cells, gather resources, or interact with one another.
- Resources are limited, encouraging both competition and cooperation.

### 🧩 AI Agents
- Each agent is powered by a small **Actor–Critic neural network** (see `agent_brain.py`).
- The network learns from experience using a replay buffer and reward feedback.
- Agents perceive local state information:
  - Their **wealth**
  - Their **reputation**
  - The **average resources** in their current cell
  - The **number of other agents** nearby
- Available actions:
  ```
  move, gather, help, donate, trade, steal
  ```

### 📊 Live Visualization (Jupyter)
The simulation visualizes three synchronized panels:
1. **Social Graph (top-left):**  
   - Nodes represent agents (size = wealth, color = points).  
   - Edges represent interactions:  
     - 🟩 **Green** = help (cooperation)  
     - 🔵 **Blue** = trade (neutral exchange)  
     - 🔴 **Red** = steal (conflict)
2. **World Map (top-right):**  
   - Shows agents’ positions on the grid.  
   - Cell color intensity (green) = wealth concentration.  
   - Agents are colored by performance points.
3. **Metrics Over Time (bottom):**  
   - Tracks global statistics across ticks:
     - Average wealth  
     - Average points  
     - Number of cooperative (“help”) edges  
     - Number of conflict (“steal”) edges  

### 🎞️ Replay Timeline
When the simulation finishes, use an **interactive slider** to replay it through time:
- Drag the slider left/right to move between frames.
- Each frame shows the society’s network, world, and agents at that specific tick.

---

## 🧰 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/society-sim.git
cd society-sim
```

### 2️⃣ Create a Python environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install torch matplotlib networkx ipywidgets
```

Enable Jupyter widgets (for the slider to work):

```bash
jupyter nbextension enable --py widgetsnbextension
```

Alternatively, create a `requirements.txt`:

```
torch
matplotlib
networkx
ipywidgets
```

and install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 How to Run the Simulation (Inside Jupyter)

### Start Jupyter

```bash
jupyter lab
```

Then open a new notebook and run:

```python
from society_sim import run_simulation

# Run simulation with 8 agents for 100 ticks
world = run_simulation(agents=8, ticks=100, pause=0.03)
```

Once complete, replay the entire timeline:

```python
world.replay()
```

You’ll see a **slider** under the plot — drag it to move through time, frame by frame 🎞️.

---

## 🧩 Interpreting the Visualization

### 🕸️ Social Graph
| Visual Element | Meaning |
|-----------------|----------|
| Node size | Agent’s wealth |
| Node color | Agent’s performance (points) |
| 🟩 Green edge | “Help” — cooperative interaction |
| 🔵 Blue edge | “Trade” — neutral, balanced exchange |
| 🔴 Red edge | “Steal” — aggressive/conflict interaction |

> The mix of edge colors reveals the overall tone of the society — cooperative or conflict-driven.

---

### 🌍 World Map
| Visual Element | Meaning |
|-----------------|----------|
| Grid cell | A geographic zone with renewable resources |
| Green intensity | Average wealth of agents in the cell |
| Colored dots | Agents (color = points, brighter = stronger) |
| Coordinates `(x, y)` | Grid position |

> Agents move, gather, or interact — creating wealth clusters and dynamic population flows.

---

### 📈 Society Metrics (Bottom Panel)
| Line | Meaning |
|------|----------|
| 🟢 Green | Average wealth of all agents |
| 🟣 Purple | Average points (performance) |
| 🔵 Blue dashed | Number of “help” edges (cooperation) |
| 🔴 Red dashed | Number of “steal” edges (conflict) |

> Rising green line = growing economy  
> Rising red dashed = conflict increasing  
> Rising blue dashed = cooperation emerging  

---

## ⚙️ Configuration and Tuning

You can modify constants in `society_sim.py` or `agent_brain.py` to experiment with different societies.

| Parameter | Description | Default |
|------------|-------------|----------|
| `HELP_REWARD` | Reward for helping others | 4.0 |
| `TRADE_REWARD` | Reward for trading | 2.0 |
| `STEAL_PENALTY` | Penalty for stealing | -6.0 |
| `regen_per_tick` | Resource regeneration speed | 2 |
| `capacity` | Max stock in each cell | 50 |
| `ticks` | Duration of simulation | 100 |

> For example, setting `STEAL_PENALTY=-10` and `HELP_REWARD=6` encourages a more cooperative civilization.

---

## 🧩 File Structure

```
society-sim/
├── agent_brain.py     # Reinforcement Learning: Actor–Critic model
├── society_sim.py     # World logic, visualization, replay system
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 🔬 Example Jupyter Workflow

```python
# 1. Import and run
from society_sim import run_simulation
world = run_simulation(agents=10, ticks=80, pause=0.05)

# 2. Replay results
world.replay()

# 3. Access world data programmatically
print("Total ticks:", world.time)
print("Agent 0 wealth:", world.agents[0].wealth)
print("Cooperative edges:", len(world.graph.edge_types["help"]))
```

---

## 🧠 How Learning Works

Each agent maintains its own neural network defined in `agent_brain.py`:

- **Actor** → decides which action to take (policy).
- **Critic** → estimates how good the current state is (value function).
- **Replay Buffer** → stores past experiences for stable training.

Agents receive *reward signals* based on their behavior:
- Helping others → positive reward  
- Stealing → penalty  
- Gathering resources → mild reward  

Over many ticks, this feedback drives emergent strategies — from pure selfishness to cooperative clusters.

---

## 🧮 Mathematical Summary

Let:
- `s_t` = current state  
- `a_t` = action  
- `r_t` = reward  
- `V(s_t)` = critic-estimated value  

Each update minimizes:
```
L = -log π(a_t | s_t) * (r_t + γV(s_{t+1}) - V(s_t)) + (r_t + γV(s_{t+1}) - V(s_t))²
```
where:
- The first term updates the policy (actor).
- The second term trains the critic to predict correct state values.

---

## 🧮 Sample Output Metrics (Example Run)

| Tick | Avg Wealth | Avg Points | Help Edges | Steal Edges |
|-------|-------------|-------------|-------------|-------------|
| 0 | 2.1 | 0.0 | 0 | 0 |
| 20 | 5.3 | 2.4 | 10 | 28 |
| 60 | 10.1 | 4.5 | 21 | 55 |
| 100 | 14.8 | 7.2 | 27 | 62 |

You’ll see this reflected in the line chart — society tends to accumulate wealth even under conflict, but cooperation improves overall stability.

---

## 🧠 Tips for Exploration

- Increase `ticks` to 500+ and observe long-term equilibria.
- Try fewer agents with high `HELP_REWARD` for utopian behavior.
- Try many agents with low resources for chaotic, competitive dynamics.
- Introduce new policies (like taxation or voting) as extra rules in `World.step()`.

---

## 💡 Possible Extensions

✅ Add a **Gini coefficient** metric to measure inequality.  
✅ Create **alliances** or **reputation systems** between agents.  
✅ Introduce **taxation** and **universal income** policies.  
✅ Export the full metrics history to CSV for analysis in Pandas.  
✅ Build an **interactive control panel** (buttons for pause/resume).  

---

## 🧠 Example: Running in Docker

You can also containerize the project for portability.

**Dockerfile**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch matplotlib networkx ipywidgets jupyter
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
```

Then build and run:

```bash
docker build -t society-sim .
docker run -p 8888:8888 society-sim
```

Follow the URL shown in the logs to access Jupyter Lab in your browser.

---

## 📜 License

**MIT License**

You are free to use, modify, and distribute this project for research, education, or creative experiments.  
Attribution is appreciated.

---

## ✨ Credits

Developed by **Paul Dubourg**  
Designed for exploratory AI research, social behavior modeling, and interactive visualization in Python.

> “A society of learning agents — where cooperation, greed, and evolution emerge from code.”

---

## 🧩 Badges

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Platform-Jupyter-orange)
![RL](https://img.shields.io/badge/Reinforcement-Learning-purple)

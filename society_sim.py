from __future__ import annotations
"""
Society Simulation — Jupyter version with live metrics, replay slider,
and clean 3-panel layout (no empty subplot).
"""

import random, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from agent_brain import BrainWrapper, ACTIONS

# Inline plotting for notebooks
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import networkx as nx
from IPython.display import clear_output, display
import numpy as np
import ipywidgets as widgets
from matplotlib.gridspec import GridSpec


# ==== Core Data ===========================================================

@dataclass
class RewardConfig:
    HELP_REWARD: float = 4.0
    TRADE_REWARD: float = 2.0
    DONATION_REWARD: float = 3.0
    GATHER_REWARD: float = 1.0
    STEAL_PENALTY: float = -6.0
    STARVATION_PENALTY: float = -3.0

@dataclass
class Policy:
    tax_rate: float = 0.1
    theft_allowed: bool = False
    regen_multiplier: float = 1.0

@dataclass
class Zone:
    name: str
    capacity: int
    stock: int
    regen_per_tick: int
    def tick(self, policy: Policy):
        regen = max(0, int(round(self.regen_per_tick * policy.regen_multiplier)))
        self.stock = min(self.capacity, self.stock + regen)

@dataclass
class Cell:
    x: int
    y: int
    zone: Zone
    occupants: List[int] = field(default_factory=list)


# ==== Visualization ======================================================

class GraphTracker:
    def __init__(self, ids: List[int]):
        self.G = nx.Graph()
        for i in ids:
            self.G.add_node(i)
        self.edge_types = {"help": set(), "trade": set(), "steal": set()}

    def log(self, kind,a,b):
        u,v = (a,b) if a<b else (b,a)
        self.G.add_edge(u,v)
        self.edge_types.setdefault(kind,set()).add((u,v))


class Visualizer:
    """Draws social graph + world map + live metrics inline in Jupyter + replay mode."""
    def __init__(self, world:'World'):
        self.world = world

        # ---- CLEAN 3-PANEL LAYOUT ----
        self.fig = plt.figure(figsize=(13, 8))
        gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], figure=self.fig)
        self.ax_graph = self.fig.add_subplot(gs[0, 0])
        self.ax_map   = self.fig.add_subplot(gs[0, 1])
        self.ax_metrics = self.fig.add_subplot(gs[1, :])
        self.fig.subplots_adjust(hspace=0.4)

        # Graph layout + color scale
        self.pos = nx.spring_layout(world.graph.G, seed=7)
        self.sm = mpl.cm.ScalarMappable(cmap=plt.cm.plasma)
        self.sm.set_clim(0,1)
        self.cbar = self.fig.colorbar(
            self.sm, ax=[self.ax_graph, self.ax_map],
            orientation='horizontal', fraction=0.05, pad=0.05
        )
        self.cbar.set_label('Agent points (color scale)', fontsize=8)

        # Historical metrics
        self.ticks = []
        self.avg_wealth = []
        self.avg_points = []
        self.coop = []
        self.conflict = []

    def update(self):
        axg, axm, axd = self.ax_graph, self.ax_map, self.ax_metrics
        axg.clear(); axm.clear(); axd.clear()

        G = self.world.graph.G
        agents = self.world.agents
        if not agents: return

        wealths={a.id:a.wealth for a in agents}
        points={a.id:a.points for a in agents}
        sizes=[120+20*wealths.get(n,1) for n in G.nodes]
        vals=list(points.values()) or [0]
        vmin,vmax=min(vals),max(vals)
        denom=vmax-vmin if vmax>vmin else 1
        colors=[(points[n]-vmin)/denom for n in G.nodes]
        self.sm.set_clim(vmin,vmax)

        # --- Social Graph ---
        nx.draw(G,self.pos,node_size=sizes,node_color=colors,
                cmap=plt.cm.plasma,ax=axg,with_labels=False)
        for kind,color in [("help","green"),("trade","blue"),("steal","red")]:
            nx.draw_networkx_edges(G,self.pos,
                edgelist=list(self.world.graph.edge_types.get(kind,[])),
                edge_color=color,width=1,ax=axg)
        axg.set_title(f"Social Graph (t={self.world.time})")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0],color='green',lw=2,label='Help'),
            Line2D([0],[0],color='blue',lw=2,label='Trade'),
            Line2D([0],[0],color='red',lw=2,label='Steal'),
            Line2D([0],[0],marker='o',color='w',markerfacecolor='purple',
                   markersize=6,label='Agent (by points)')
        ]
        axg.legend(handles=legend_elements,loc='lower right',fontsize=8,frameon=True)

        # --- World Map ---
        cells=self.world.cells
        xs=[c.x for c in cells.values()]; ys=[c.y for c in cells.values()]
        minx,maxx,miny,maxy=min(xs)-1,max(xs)+1,min(ys)-1,max(ys)+1
        axm.set_xlim(minx,maxx+1); axm.set_ylim(miny,maxy+1); axm.invert_yaxis()

        for cell in cells.values():
            avgw=(sum(self.world.get_agent(a).wealth for a in cell.occupants)/len(cell.occupants)) if cell.occupants else 0
            rect=patches.Rectangle((cell.x,cell.y),1,1,
                                   facecolor=plt.cm.Greens(min(avgw/10,1)),
                                   edgecolor='black',lw=0.5)
            axm.add_patch(rect)
            for aid in cell.occupants:
                a=self.world.get_agent(aid)
                cval=(a.points-vmin)/denom
                axp=cell.x+0.2+random.random()*0.6
                ayp=cell.y+0.2+random.random()*0.6
                axm.plot(axp,ayp,'o',color=plt.cm.plasma(cval),markersize=4)
            axm.text(cell.x+0.5,cell.y+0.9,f"({cell.x},{cell.y})",
                     ha='center',va='top',fontsize=7)
        axm.set_title("World Map")
        axm.axis('off')

        import matplotlib.patches as mpatches
        wealth_patch = mpatches.Patch(color=plt.cm.Greens(0.6), label='Cell wealth')
        agent_patch = Line2D([0],[0],marker='o',color='w',markerfacecolor='purple',
                             markersize=6,label='Agent (by points)')
        axm.legend(handles=[wealth_patch,agent_patch],loc='lower right',fontsize=8,frameon=True)
        self.cbar.update_normal(self.sm)

        # --- Metrics ---
        t = self.world.time
        self.ticks.append(t)
        self.avg_wealth.append(np.mean([a.wealth for a in agents]))
        self.avg_points.append(np.mean([a.points for a in agents]))
        self.coop.append(len(self.world.graph.edge_types.get("help",[])))
        self.conflict.append(len(self.world.graph.edge_types.get("steal",[])))

        axd.plot(self.ticks, self.avg_wealth, label="Avg wealth", color="green")
        axd.plot(self.ticks, self.avg_points, label="Avg points", color="purple")
        axd.plot(self.ticks, self.coop, label="Help edges", color="blue", linestyle="--")
        axd.plot(self.ticks, self.conflict, label="Steal edges", color="red", linestyle="--")
        axd.set_title("Society Metrics Over Time")
        axd.set_xlabel("Ticks")
        axd.legend(fontsize=8, ncol=2)
        axd.grid(True, linestyle=":", alpha=0.5)

        clear_output(wait=True)
        display(self.fig)
        plt.close(self.fig)

    # --- Draw a static snapshot for replay ---
    def draw_snapshot(self, frame):
        fig, (axg, axm) = plt.subplots(1, 2, figsize=(12, 5))
        G = self.world.graph.G
        vals = frame["points"]
        vmin, vmax = min(vals), max(vals)
        denom = vmax - vmin if vmax > vmin else 1
        colors = [(v - vmin)/denom for v in vals]

        nx.draw(G, self.pos, node_size=100, node_color=colors,
                cmap=plt.cm.plasma, ax=axg, with_labels=False)
        for kind, color in [("help","green"),("trade","blue"),("steal","red")]:
            nx.draw_networkx_edges(G, self.pos, edgelist=frame["edges"].get(kind, []),
                                   edge_color=color, width=1, ax=axg)
        axg.set_title(f"Social Graph (t={frame['time']})")

        for (aid,(x,y)) in frame["positions"].items():
            axm.plot(x+0.5, y+0.5, 'o',
                     color=plt.cm.plasma((frame["points"][aid]-vmin)/denom),
                     markersize=5)
        axm.set_title("World Map Snapshot")
        axm.axis("equal"); axm.axis("off")

        display(fig)
        plt.close(fig)


# ==== Agents =============================================================

class Agent:
    def __init__(self,id:int,name:str,wealth:int=3):
        self.id=id; self.name=name; self.wealth=wealth
        self.reputation=0.0; self.points=0.0
        self.brain=BrainWrapper(state_size=4)
        self.last_state=None

    def perceive(self,world:'World'):
        x,y=world.agent_positions[self.id]
        cell=world.get_or_create_cell(x,y)
        n=len(cell.occupants); avg_stock=cell.zone.stock/cell.zone.capacity
        return torch.tensor([[self.wealth/10,self.reputation,avg_stock,n/10]],dtype=torch.float32)

    def step(self,world:'World'):
        s=self.perceive(world)
        if self.last_state is None: self.last_state=s
        action=self.brain.select_action(s)
        self.brain.last_state=s
        old_points=self.points
        self.execute(action,world)
        reward=self.points-old_points
        ns=self.perceive(world)
        self.brain.learn(ns,reward,done=False)

    def execute(self,action:str,world:'World'):
        if action=="move": self.move(world)
        elif action=="gather": self.gather(world)
        elif action=="help": self.help_someone(world)
        elif action=="donate": self.donate(world)
        elif action=="trade": self.trade(world)
        elif action=="steal": self.steal(world)
        else: self.points+=0.01

    def move(self,world):
        x,y=world.agent_positions[self.id]
        dx,dy=random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        world.move_agent(self.id,x+dx,y+dy)
    def gather(self,world):
        c=world.get_or_create_cell(*world.agent_positions[self.id])
        amt=min(c.zone.stock, random.randint(1,3))
        c.zone.stock-=amt; self.wealth+=amt; self.points+=0.1
    def help_someone(self,world):
        others=[a for a in world.agents if a.id!=self.id]
        if others and self.wealth>1:
            t=min(others,key=lambda a:a.wealth)
            self.wealth-=1; t.wealth+=1; self.points+=0.2
            world.graph.log("help",self.id,t.id)
    def donate(self,world):
        self.points+=0.1; self.wealth=max(0,self.wealth-1)
    def trade(self,world):
        others=[a for a in world.agents if a.id!=self.id]
        if others and self.wealth>0:
            p=random.choice(others); self.points+=0.15
            world.graph.log("trade",self.id,p.id)
    def steal(self,world):
        victims=[a for a in world.agents if a.id!=self.id and a.wealth>0]
        if victims:
            v=random.choice(victims)
            self.wealth+=1; v.wealth-=1; self.points-=0.05
            world.graph.log("steal",self.id,v.id)


# ==== World ==============================================================

@dataclass
class World:
    agents:List[Agent]; policy:Policy; rewards:RewardConfig
    graph:GraphTracker=field(init=False)
    cells:Dict[Tuple[int,int],Cell]=field(default_factory=dict)
    agent_positions:Dict[int,Tuple[int,int]]=field(default_factory=dict)
    time:int=0; viz:Optional[Visualizer]=None
    history:List[Dict]=field(default_factory=list)

    def __post_init__(self):
        self.graph=GraphTracker([a.id for a in self.agents])
        self.get_or_create_cell(0,0)
        for a in self.agents:
            self.agent_positions[a.id]=(0,0)
            self.cells[(0,0)].occupants.append(a.id)

    def get_or_create_cell(self,x:int,y:int)->Cell:
        if (x,y) in self.cells: return self.cells[(x,y)]
        z=Zone('Field',50,25,2)
        c=Cell(x,y,z); self.cells[(x,y)]=c; return c
    def get_agent(self,aid:int)->Agent:
        for a in self.agents:
            if a.id==aid: return a
    def move_agent(self,aid:int,x:int,y:int):
        ox,oy=self.agent_positions[aid]
        if (ox,oy) in self.cells and aid in self.cells[(ox,oy)].occupants:
            self.cells[(ox,oy)].occupants.remove(aid)
        c=self.get_or_create_cell(x,y); c.occupants.append(aid)
        self.agent_positions[aid]=(x,y)
    def step(self):
        self.time+=1
        for a in self.agents: a.step(self)
        if self.time%20==0:
            for a in self.agents: a.brain.replay_train(batch_size=16)
        # record snapshot for replay
        snapshot = {
            "time": self.time,
            "positions": dict(self.agent_positions),
            "points": [a.points for a in self.agents],
            "wealth": [a.wealth for a in self.agents],
            "edges": {k: list(v) for k,v in self.graph.edge_types.items()}
        }
        self.history.append(snapshot)
        if self.viz: self.viz.update()

    # --- Replay the simulation interactively ---
    def replay(self):
        if not self.history:
            print("No history recorded.")
            return
        slider = widgets.IntSlider(min=0, max=len(self.history)-1, step=1, value=0, description='Tick')
        output = widgets.Output()

        def on_value_change(change):
            i = change['new']
            frame = self.history[i]
            clear_output(wait=True)
            self.viz.draw_snapshot(frame)
            display(slider)

        slider.observe(on_value_change, names='value')
        display(slider)
        self.viz.draw_snapshot(self.history[0])


# ==== Runner ==============================================================

def build_world(n_agents:int,seed=None)->World:
    if seed is not None: random.seed(seed)
    agents=[Agent(i,f"Agent{i}",wealth=random.randint(1,10)) for i in range(n_agents)]
    return World(agents,Policy(),RewardConfig())

def run_simulation(agents=6,ticks=100,seed=None,pause=0.05):
    world=build_world(agents,seed)
    world.viz=Visualizer(world)
    for _ in range(ticks):
        world.step()
        time.sleep(pause)
    return world


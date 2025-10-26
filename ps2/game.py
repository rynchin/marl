from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import lru_cache

@dataclass
class Node:
    history: str
    kind: str # 'terminal', 'chance', 'decision'
    player: Optional[int]
    actions: List[str]
    probs: Optional[List[float]] = None # 'chance'
    payoff: Optional[float] = None # 'terminal'

class Game:
    def __init__(self, filename: str):
        self.nodes: Dict[str, Node] = {}
        self.children: Dict[Tuple[str, str], str] = {} # (history, action) -> child_history
        self.infosets: Dict[str, List[str]] = {} # info_set -> list of histories
        self.infoset_of: Dict[str, str] = {} # history -> info_set name
        self.acts_at_infoset: Dict[str, List[str]] = {} # info_set -> list of actions
        self.player_at_infoset: Dict[str, int] = {} # info_set -> player
        self.load_game(filename)
        self.build_tree()
        self.build_infosets()
    
    def load_game(self, filename: str):
        """
        Load the game from filename
        """
        with open(filename, 'r') as file:
            lines = file.read().strip().splitlines()
            lines = [ln for ln in lines if ln.strip()]
    
        for line in lines:
            if line.startswith("node "):
                parts = line.split()
                history = parts[1]
                kind = parts[2]
                if kind == "terminal": # terminal node
                    # game is zero-sum
                    # node history terminal payoffs 1=... 2=...
                    p1 = None
                    for tok in parts:
                        if tok.startswith("1="):
                            p1 = float(tok[2:])
                            break
                    self.nodes[history] = Node(history, "terminal", None, [], None, p1)
                elif kind == "chance": # chance node
                    # node history chance actions a1=... a2=... ...
                    idx = parts.index("actions")
                    actions, probs = [], []
                    for tok in parts[idx+1:]:
                        a, p = tok.split("=")
                        actions.append(a)
                        probs.append(float(p))
                    self.nodes[history] = Node(history, "chance", None, actions, probs, None)
                elif kind == "player": # decision
                    # node history player <X> actions a1 a2 ...
                    player = int(parts[3])
                    idx = parts.index("actions")
                    actions = parts[idx+1:]
                    self.nodes[history] = Node(history, "decision", player, actions, None, None)
                else:
                    raise ValueError(f"Invalid node: {line}")
            elif line.startswith("infoset "):
                # infoset <NAME> nodes <N1> ... <Nn>
                parts = line.split()
                name = parts[1]
                idx = parts.index("nodes")
                histories = parts[idx+1:]
                self.infosets[name] = histories # info_set -> list of histories
                for h in histories:
                    self.infoset_of[h] = name # map history to info_set name


    def build_tree(self):
        """
        Build the tree of the game.
        """
        children: Dict[str, List[str]] = {} # history -> list of children histories
        for node in self.nodes:
            if node == "/": # root
                continue
            subsections = [s for s in node.split("/") if s]
            parent = "/" if len(subsections) == 1 else "/" + "/".join(subsections[:-1]) + "/" # merge path
            if parent not in children:
                children[parent] = []
            children[parent].append(node)
        for parent, node in self.nodes.items():
            if node.kind == "terminal":
                continue
            for child in children.get(parent, []):
                last_action = [s for s in child.split("/") if s][-1]
                if ":" in last_action: # player: action
                    action = last_action.split(":")[1]
                    self.children[(parent, action)] = child
    
    def build_infosets(self):
        """
        Build the infosets of the game.
        """
        for name, histories in self.infosets.items():
            n = self.nodes[histories[0]]
            self.acts_at_infoset[name] = n.actions
            self.player_at_infoset[name] = n.player
            
    # ------------------------------------------------------------
    # Strats
    # ------------------------------------------------------------

    def uniform(self):
        """
        Uniform strategy for each player
        """
        s1, s2 = {}, {}
        for name, actions in self.acts_at_infoset.items():
            probs = [1.0 / len(actions)] * len(actions)
            if self.player_at_infoset[name] == 1:
                s1[name] = probs
            else:
                s2[name] = probs
        return s1, s2
    
    def value(self, s1: Dict[str, List[float]], s2: Dict[str, List[float]]):
        """
        Value of the game
        """
        # convert strategy profiles to numpy arrays
        s1_info = {name: np.asarray(probs) for name, probs in s1.items()}
        s2_info = {name: np.asarray(probs) for name, probs in s2.items()}

        @lru_cache(maxsize=None)
        def val(history: str) -> float:
            n = self.nodes[history] # get node from history
            if n.kind == "terminal": # terminal node
                return n.payoff
            if n.kind == "chance": # chance node
                return sum(val(self.children[(history, a)]) * p for a, p in zip(n.actions, n.probs))
            infoset_name = self.infoset_of[history]
            if n.player == 1:
                return sum(val(self.children[(history, a)]) * s1_info[infoset_name][i] for i, a in enumerate(n.actions))
            else:
                return sum(val(self.children[(history, a)]) * s2_info[infoset_name][i] for i, a in enumerate(n.actions))
        
        # calculate from root
        return val("/")

    def best_response(self, player: int, opponent_strategy: Dict[str, List[float]]):
        """
        Best response for player to opponent's strategy.
        Maximize sum_{h in I} pi_{-i}(h) * V(child(h,a))
        """
        os_info = {name: np.asarray(probs) for name, probs in opponent_strategy.items()}
        best_response = {} # Dict[str, np.ndarray], will map info_set name to best response strategy
        
        # opponent reach probabilities
        reach = {"/": 1.0}
        stack = ["/"]
        while stack:
            history = stack.pop()
            n = self.nodes[history] # get node from history
            if n.kind == "terminal": # terminal node
                continue
            if n.kind == "chance": # chance node
                for a, p in zip(n.actions, n.probs):
                    child = self.children[(history, a)]
                    reach[child] = reach[history] * p
                    stack.append(child)
            else:
                if n.player != player: # decision node of opponent
                    name = self.infoset_of[history] # infoset name
                    distrib = os_info[name]
                    for idx, a in enumerate(n.actions):
                        child = self.children[(history, a)]
                        reach[child] = reach[history] * distrib[idx]
                        stack.append(child)
                else: # decision node of player
                    for a in n.actions:
                        child = self.children[(history, a)]
                        reach[child] = reach[history]
                        stack.append(child)

        decided = set() # if best response is decided, stash it

        @lru_cache(maxsize=None)
        def val(history: str) -> float:
            n = self.nodes[history] # get node from history
            if n.kind == "terminal": # terminal node
                return n.payoff if player == 1 else -n.payoff
            if n.kind == "chance": # chance node
                return sum(val(self.children[(history, a)]) * p for a, p in zip(n.actions, n.probs))
            if n.player == player:
                name = self.infoset_of[history] # infoset name
                if name not in decided:
                    actions = self.acts_at_infoset[name]
                    expected_values = []
                    for a in actions:
                        sumer = 0
                        for hist_name in self.infosets[name]:
                            child = self.children[(hist_name, a)]
                            sumer += reach.get(child, 0) * val(child)
                        expected_values.append(sumer)

                    best_val_idx = np.argmax(expected_values) # get best value index
                    br = np.zeros(len(actions))
                    br[best_val_idx] = 1.0
                    best_response[name] = br
                    decided.add(name)
                
                # use chosen action
                idx = np.argmax(best_response[name])
                action = self.acts_at_infoset[name][idx]
                return val(self.children[(history, action)])
            else:
                name = self.infoset_of[history] # infoset name
                vals = [val(self.children[(history, a)]) for a in n.actions]
                return np.dot(os_info[name], np.asarray(vals))
        
        # calculate from root
        val = val("/")
        player_br = {name: strat for name, strat in best_response.items() if player == self.player_at_infoset[name]}
        if player == 1:
            return val, player_br
        else:
            return -val, player_br

    def nash_gap(self, x,y):
        """
        Nash gap for P1
        """
        v1, s1_br = self.best_response(1, y)
        v2, s2 = self.best_response(2, x)
        s2_val = {k: np.asarray(s2.get(k, v)) for k, v in y.items()}
        return v1 - self.value(x, s2_val)
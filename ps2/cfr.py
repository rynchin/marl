from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from game import Game

# Regret Matching class
@dataclass
class RegretMatching:
    actions: Tuple[str, ...]
    regret: np.ndarray = field(init=False)
    strategy_sum: np.ndarray = field(init=False)

    def __post_init__(self): # initialize empty regret and strategy sum
        n = len(self.actions)
        self.regret = np.zeros(n)
        self.strategy_sum = np.zeros(n)

    def current_strategy(self) -> np.ndarray:
        regrets = np.maximum(self.regret, 0.0)
        sumer = regrets.sum()
        if sumer > 1e-15: # avoid division by zero
            return regrets / sumer
        else:
            return np.full(regrets.size, 1.0 / regrets.size) # uniform

    def add_strategy_mass(self, sigma: np.ndarray, weight: float) -> None:
        if weight > 0:
            self.strategy_sum += weight * sigma

    def average_strategy(self) -> np.ndarray:
        sumer = self.strategy_sum.sum()
        if sumer > 0:
            return self.strategy_sum / sumer
        else:
            return np.full(self.strategy_sum.size, 1.0 / self.strategy_sum.size) # uniform

    def cfr_update_regret(self, util_a: np.ndarray, util: float, opp_reach: float) -> None:
        # regret = opp_reach * (action utility - strategy utility)
        if opp_reach:
            self.regret += opp_reach * (util_a - util) 

# CFR Trainer class
class CFRTrainer:
    def __init__(self, game: Game):
        self.game = game
        self.rm = {1: {}, 2: {}} # regret matching for each player
        for name, actions in game.acts_at_infoset.items():
            player = game.player_at_infoset[name]
            self.rm[player][name] = RegretMatching(tuple(actions)) # rm for each player at each infoset

    def train_uniform(self, T: int): # P1 vs uniform P2
        _, y = self.game.uniform() # P2 fixed uniform
        util_history = []
        for _ in range(T):
            self.cfr(
                history="/", 
                pi1=1.0, 
                pi2=1.0, 
                pic=1.0, 
                update_player=1, 
                fixed_sigma_2=y
            )

            x_avg = self.get_average_strategy(1)

            util_history.append(self.game.value(x_avg, y)) # utility
        return self.get_average_strategy(1), y, util_history # strategy for P1, fixed P2, utility history

    def train_both(self, T: int): # both players
        util_history, gap_history = [], []
        for _ in range(T):
            self.cfr("/", 
                pi1=1.0, 
                pi2=1.0, 
                pic=1.0, 
                update_player=0, 
                fixed_sigma_2=None
            )

            x_avg = self.get_average_strategy(1)
            y_avg = self.get_average_strategy(2)

            util_history.append(self.game.value(x_avg, y_avg)) # utility
            gap_history.append(self.game.nash_gap(x_avg, y_avg))
        return self.get_average_strategy(1), self.get_average_strategy(2), util_history, gap_history # strategy for P1, strategy for P2, utility, gap

    def get_average_strategy(self, player: int):
        return {name: self.rm[player][name].average_strategy().tolist() for name in self.rm[player]} # average strategy at each infoset

    def cfr(
        self,
        history: str,
        pi1: float, # P1's action probability
        pi2: float, # P2's action probability
        pic: float, # chance node probability
        update_player: int, # 0=both, 1=only P1, 2=only P2
        fixed_sigma_2: Optional[Dict[str, List[float]]] # only for fixed P2
    ):
        node = self.game.nodes[history]

        if node.kind == "terminal":
            return node.payoff

        if node.kind == "chance":
            util = 0.0
            for a, p in zip(node.actions, node.probs):
                child = self.game.children[(history, a)]
                util += p * self.cfr(child, pi1, pi2, pic * p, update_player, fixed_sigma_2)
            return util

        # decision node
        player = node.player
        name = self.game.infoset_of[history]
        actions = self.game.acts_at_infoset[name]

        # strategy at infoset
        if player == 2 and fixed_sigma_2 is not None:
            sigma = np.asarray(fixed_sigma_2[name], dtype=float)
        else:
            sigma = self.rm[player][name].current_strategy()

        # recursive action utilities
        util_a = np.zeros(len(actions))
        for i, act in enumerate(actions):
            child = self.game.children[(history, act)]
            if player == 1: # P1's action
                util_a[i] = self.cfr(child, pi1 * sigma[i], pi2, pic, update_player, fixed_sigma_2)
            else: # P2's action
                util_a[i] = self.cfr(child, pi1, pi2 * sigma[i], pic, update_player, fixed_sigma_2)

        util = float(np.dot(sigma, util_a))

        # update regret
        if player == 1: # P1's action
            if update_player in (0, 1):
                self.rm[1][name].cfr_update_regret(util_a, util, opp_reach = pi2 * pic)
            self.rm[1][name].add_strategy_mass(sigma, weight = pi1 * pic)
        else:
            if fixed_sigma_2 is None: # P2's action
                if update_player in (0, 2):
                    self.rm[2][name].cfr_update_regret(-util_a, -util, opp_reach = pi1 * pic) # negate due to zero-sum
                self.rm[2][name].add_strategy_mass(sigma, weight = pi2 * pic)
            else:
                self.rm[2][name].add_strategy_mass(sigma, weight = pi2 * pic)

        return util

# Problem 5.2 (Uniform)
def run_problem_52(files, T: int = 1000) -> None:
    for name, path in files.items():
        game = Game(path)
        tr = CFRTrainer(game)
        x, y, util_history = tr.train_uniform(T)
        plt.figure()
        plt.plot(range(1, T + 1), util_history)
        plt.xlabel("Iterations T")
        plt.ylabel("Utility $u_1(x_T, y)$")
        plt.title(f"CFR (Uniform) — {name}")
        plt.tight_layout()
        out_utility = f"images/uniform/cfr_{name}.png"
        plt.savefig(out_utility)
        plt.close()
        print(f"Problem 5.2 - {name}: final u1 = {util_history[-1]:.6f}")

# Problem 5.3 (Both)
def run_problem_53(files, T: int = 1000) -> None:
    for name, path in files.items():
        game = Game(path)
        tr = CFRTrainer(game)
        x, y, util_history, gap_history = tr.train_both(T)
        plt.figure()
        plt.plot(range(1, T + 1), util_history)
        plt.xlabel("Iterations T")
        plt.ylabel("Utility $u_1(x_T, y_T)$")
        plt.title(f"CFR (Both) — {name}")
        plt.tight_layout()
        out_utility = f"images/both/cfr_{name}.png"
        plt.savefig(out_utility)
        plt.close()

        plt.figure()
        plt.plot(range(1, T + 1), gap_history)
        plt.xlabel("Iterations T")
        plt.ylabel("Nash Gap $\gamma(x_T, y_T)$")
        plt.title(f"CFR - Nash Gap (Both) — {name}")
        plt.tight_layout()
        out_nash_gap = f"images/both/cfr_nash_gap_{name}.png"
        plt.savefig(out_nash_gap)
        plt.close()

        print(f"Problem 5.3 - {name}: final u1 = {util_history[-1]:.6f}, gap = {gap_history[-1]:.6f}")

if __name__ == "__main__":
    FILES = {
        "rpss": "efgs/rock_paper_superscissors.txt",
        "kuhn": "efgs/kuhn.txt",
        "leduc2": "efgs/leduc2.txt",
    }
    run_problem_52(FILES, 1000)
    run_problem_53(FILES, 1000)
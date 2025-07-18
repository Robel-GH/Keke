from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple, Set
import heapq
import math
import itertools
import time


class ASTAR2Agent(BaseAgent):
    def search(self, initial_state: GameState, iterations: int = 1000, timeout: float = 60.0) -> List[Direction]:
        frontier = []
        visited = set()
        counter = itertools.count()
        start_time = time.time()

        initial_h = self._heuristic(initial_state)
        heapq.heappush(frontier, (initial_h, 0, next(counter), initial_state, []))

        while frontier:
            # Controllo timeout
            if time.time() - start_time > timeout:
                print(f"[TIMEOUT] Superato il limite di {timeout} secondi")
                return []

            f_score, g_score, _, current_state, path = heapq.heappop(frontier)

            state_key = str(current_state)
            if state_key in visited:
                continue
            visited.add(state_key)

            if check_win(current_state):
                return path

            if g_score >= iterations:
                continue

            for action in self._ordered_actions(current_state):
                next_state = advance_game_state(action, current_state.copy())
                next_path = path + [action]
                next_g = g_score + 1
                next_h = self._heuristic(next_state)
                next_f = next_g + next_h
                heapq.heappush(frontier, (next_f, next_g, next(counter), next_state, next_path))

        return []
    
    def _heuristic(self, state: GameState) -> int:
        """
        Euristica: distanza di Manhattan + penalità per ostacoli.
        """
        min_cost = math.inf
        for player in state.players:
            for win_obj in state.winnables:
                dx = abs(player.x - win_obj.x)
                dy = abs(player.y - win_obj.y)
                base_distance = dx + dy

                obstacle_penalty = self._count_obstacles_between(player, win_obj, state)
                cost = base_distance + obstacle_penalty

                if cost < min_cost:
                    min_cost = cost

        return min_cost if min_cost != math.inf else 0

    def _count_obstacles_between(self, player, win_obj, state: GameState) -> int:
        """
        Conta quanti ostacoli `stop`, `push` o `kill` ci sono nel rettangolo che contiene player e winnable.
        """
        x_min, x_max = min(player.x, win_obj.x), max(player.x, win_obj.x)
        y_min, y_max = min(player.y, win_obj.y), max(player.y, win_obj.y)

        penalty = 0
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                obj = state.obj_map[y][x]
                if obj == ' ':
                    continue
                if obj.is_stopped:
                    penalty += 3  # muro
                elif obj in state.pushables:
                    penalty += 2  # oggetti pushabili
                elif obj in state.killers:
                    penalty += 5  # kill objects
        return penalty

    def _ordered_actions(self, state: GameState) -> List[Direction]:
        # Potremmo ordinare le azioni per quelle che riducono di più l'euristica
        # Per semplicità ritorniamo ordine fisso ora
        return [Direction.Up, Direction.Down, Direction.Left, Direction.Right]

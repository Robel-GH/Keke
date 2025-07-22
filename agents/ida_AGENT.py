from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple
import math

class IDAAgent(BaseAgent):
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        threshold = self._heuristic(initial_state)
        path = []

        while True:
            visited = set()
            temp = self._search(initial_state, 0, threshold, path, visited)
            if isinstance(temp, list):  # soluzione trovata
                return temp
            if temp == math.inf:
                return []  # fallimento
            threshold = temp  # aumento la soglia

    def _search(self, state: GameState, g: int, threshold: float, path: List[Direction], visited: set) -> Tuple[float, List[Direction]]:
        f = g + self._heuristic(state)
        if f > threshold:
            return f
        if check_win(state):
            return path

        state_key = str(state)
        if state_key in visited:
            return math.inf
        visited.add(state_key)

        minimum = math.inf
        for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
            next_state = advance_game_state(action, state.copy())
            result = self._search(next_state, g + 1, threshold, path + [action], visited)
            if isinstance(result, list):
                return result  # soluzione trovata
            if result < minimum:
                minimum = result

        return minimum

    def _heuristic(self, state: GameState) -> int:
        min_distance = math.inf
        for player in state.players:
            for win_obj in state.winnables:
                distance = abs(player.x - win_obj.x) + abs(player.y - win_obj.y)
                if distance < min_distance:
                    min_distance = distance
        return min_distance if min_distance != math.inf else 0

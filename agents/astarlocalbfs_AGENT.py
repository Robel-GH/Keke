from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Tuple
import heapq
import math
import itertools
import time
from collections import deque

class ASTARLOCALBFSAgent(BaseAgent):
    def search(self, initial_state: GameState, iterations: int = 1000, timeout: float = 15.0) -> List[Direction]:
        frontier = []
        counter = itertools.count()
        cost_so_far = {}
        start_time = time.time()

        initial_h = self._heuristic_bfs_local(initial_state)
        initial_state_key = str(initial_state)
        heapq.heappush(frontier, (initial_h, 0, next(counter), initial_state, []))
        cost_so_far[initial_state_key] = 0

        while frontier:
            if time.time() - start_time > timeout:
                print(f"[TIMEOUT] A* con BFS locale superato il limite di {timeout}s.")
                return []

            f_score, g_score, _, current_state, path = heapq.heappop(frontier)

            state_key = str(current_state)
            if check_win(current_state):
                return path

            if g_score >= iterations:
                continue

            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(action, current_state.copy())
                next_state_key = str(next_state)
                next_g = g_score + 1

                if next_state_key not in cost_so_far or next_g < cost_so_far[next_state_key]:
                    cost_so_far[next_state_key] = next_g
                    next_h = self._heuristic_bfs_local(next_state)
                    next_f = next_g + next_h
                    next_path = path + [action]
                    heapq.heappush(frontier, (next_f, next_g, next(counter), next_state, next_path))

        return []

    def _heuristic_bfs_local(self, state: GameState, max_steps: int = 10) -> int:
        """
        Euristica: distanza minima dal winnable usando una BFS locale limitata.
        """
        min_distance = math.inf
        for player in state.players:
            distance = self._bfs_to_winnable(state, player.x, player.y, max_steps)
            if distance < min_distance:
                min_distance = distance
        return min_distance if min_distance != math.inf else 50  # penalità se irraggiungibile

    def _bfs_to_winnable(self, state: GameState, start_x: int, start_y: int, max_steps: int) -> int:
        """
        BFS limitata che cerca un winnable a partire da (start_x, start_y).
        """
        visited = set()
        queue = deque()
        queue.append((start_x, start_y, 0))

        while queue:
            x, y, steps = queue.popleft()
            if steps > max_steps:
                continue

            if (x, y) in visited:
                continue
            visited.add((x, y))

            if any(win_obj.x == x and win_obj.y == y for win_obj in state.winnables):
                return steps  # trovato un winnable

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < len(state.obj_map) and 0 <= nx < len(state.obj_map[0]):
                    obj = state.obj_map[ny][nx]
                    if obj == ' ' or not obj.is_stopped:
                        queue.append((nx, ny, steps + 1))

        return math.inf  # nessun winnable trovato entro max_steps

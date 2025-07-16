from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObjectType
from typing import List
import heapq

class KEKEAgent(BaseAgent):
    def search(self, initial_state: GameState, iterations: int = 100) -> List[Direction]:
        """Perform A* search to find a solution path."""
        open_list = []
        counter = 0
        h = self.heuristic(initial_state)
        heapq.heappush(open_list, (h, counter, initial_state, []))
        visited = set()
        
        while open_list:
            f, _, state, path = heapq.heappop(open_list)
            state_str = str(state)
            if state_str in visited:
                continue
            visited.add(state_str)
            if check_win(state):
                return path
            if len(path) >= iterations:
                continue
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
                next_state = advance_game_state(action, state.copy())
                if next_state is not None:
                    next_state_str = str(next_state)
                    if next_state_str not in visited:
                        g = len(path) + 1
                        h = self.heuristic(next_state)
                        f = g + h
                        counter += 1
                        heapq.heappush(open_list, (f, counter, next_state, path + [action]))
        return []

    def heuristic(self, state: GameState) -> float:
        """Estimate the cost to reach the goal, considering rule formation and navigation."""
        if check_win(state):
            return 0

        # Check if a "YOU" rule exists
        has_you_rule = any("IS YOU" in rule for rule in state.rules)
        if not has_you_rule:
            possible_you_rules = self.get_possible_rules(state, "YOU")
            if not possible_you_rules:
                return float('inf')
            return min(self.rule_formation_cost(state, X, "IS", "YOU") for X in possible_you_rules)

        # Check if a "WIN" rule exists
        has_win_rule = any("IS WIN" in rule for rule in state.rules)
        if has_win_rule:
            if not state.players or not state.winnables:
                return float('inf')
            return min(abs(player.x - win.x) + abs(player.y - win.y) 
                       for player in state.players 
                       for win in state.winnables)

        # Need to form a "Y IS WIN" rule
        possible_win_rules = self.get_possible_rules(state, "WIN")
        if not possible_win_rules:
            return float('inf')
        min_cost = float('inf')
        for Y in possible_win_rules:
            formation_cost = self.rule_formation_cost(state, Y, "IS", "WIN")
            Y_objs = [obj for obj in state.phys if obj.name == Y]
            if not Y_objs:
                continue
            min_dist_to_Y = min(abs(player.x - y_obj.x) + abs(player.y - y_obj.y) 
                               for player in state.players 
                               for y_obj in Y_objs)
            total_cost = formation_cost + min_dist_to_Y
            min_cost = min(min_cost, total_cost)
        return min_cost

    def get_possible_rules(self, state: GameState, property: str) -> List[str]:
        """Identify objects that can form a rule with the given property."""
        possible = set()
        for obj in state.phys:
            if obj.object_type == GameObjectType.Word:
                rule = f"{obj.name.upper()} IS {property}"
                if all(self.text_block_exists(state, word) for word in [obj.name, "IS", property]):
                    possible.add(obj.name)
        return list(possible)

    def text_block_exists(self, state: GameState, word: str) -> bool:
        """Check if a text block with the given word exists."""
        return any(obj.name == word and obj.object_type == GameObjectType.Word 
                   for obj in state.phys)

    def rule_formation_cost(self, state: GameState, A: str, B: str, C: str) -> float:
        """Estimate the cost to align text blocks A, B, C into a rule."""
        pos_A = [(obj.x, obj.y) for obj in state.phys 
                 if obj.name == A and obj.object_type == GameObjectType.Word]
        pos_B = [(obj.x, obj.y) for obj in state.phys 
                 if obj.name == B and obj.object_type == GameObjectType.Word]
        pos_C = [(obj.x, obj.y) for obj in state.phys 
                 if obj.name == C and obj.object_type == GameObjectType.Word]
        if not pos_A or not pos_B or not pos_C:
            return float('inf')
        pos_A, pos_B, pos_C = pos_A[0], pos_B[0], pos_C[0]
        
        min_cost = float('inf')
        grid_width = len(state.obj_map[0])
        grid_height = len(state.obj_map)
        
        # Horizontal alignments
        for y in range(grid_height):
            for x in range(grid_width - 2):
                cost = (abs(pos_A[0] - x) + abs(pos_A[1] - y) +
                        abs(pos_B[0] - (x+1)) + abs(pos_B[1] - y) +
                        abs(pos_C[0] - (x+2)) + abs(pos_C[1] - y))
                min_cost = min(min_cost, cost)
        
        # Vertical alignments
        for x in range(grid_width):
            for y in range(grid_height - 2):
                cost = (abs(pos_A[0] - x) + abs(pos_A[1] - y) +
                        abs(pos_B[0] - x) + abs(pos_B[1] - (y+1)) +
                        abs(pos_C[0] - x) + abs(pos_C[1] - (y+2)))
                min_cost = min(min_cost, cost)
        
        return min_cost

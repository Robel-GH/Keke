import sys
import os
from collections import deque
import copy
import heapq

# Add the parent directory to the sys.path to allow imports from baba.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from baba import GameState, Direction, advance_game_state, check_win, parse_map, make_level, character_to_name, name_to_character, is_word, is_key_word, is_phys, GameObjectType, double_map_to_string

class KEKEAgent:
    def __init__(self):
        pass

    def _get_hashable_state(self, game_state: GameState):
        """
        Converts a GameState object into a hashable representation.
        This representation includes the object map, background map, and active rules.
        """
        # Convert object map to a tuple of tuples of object names for hashing
        obj_map_hashable = tuple(tuple(obj.name if hasattr(obj, 'name') else obj for obj in row) for row in game_state.obj_map)
        back_map_hashable = tuple(tuple(obj.name if hasattr(obj, 'name') else obj for obj in row) for row in game_state.back_map)
        rules_tuple = tuple(sorted(game_state.rules))
        return (obj_map_hashable, back_map_hashable, rules_tuple)

    def _get_object_name(self, obj):
        """
        Helper function to get object name, handling both string and object types.
        """
        if obj is None:
            return None
        elif isinstance(obj, str):
            return obj
        elif hasattr(obj, 'name'):
            return obj.name
        else:
            return str(obj)

    def _get_object_type(self, obj):
        """
        Helper function to get object type, handling both string and object types.
        """
        if obj is None:
            return None
        elif isinstance(obj, str):
            # For strings, we need to infer the type based on the name
            if obj in ["is", "and"]:
                return GameObjectType.Keyword
            elif is_word(obj):
                return GameObjectType.Word
            else:
                return GameObjectType.Physical
        elif hasattr(obj, 'object_type'):
            return obj.object_type
        else:
            return None

    def _heuristic(self, game_state: GameState) -> int:
        """
        A sophisticated rule-aware heuristic.
        Combines Manhattan distance to winnable, rule formation potential, and accessibility.
        """
        h_cost = 0

        # 1. Player-to-Goal Distance (Manhattan Distance)
        min_dist_to_winnable = float("inf")
        if game_state.players and game_state.winnables:
            for player in game_state.players:
                for winnable in game_state.winnables:
                    dist = abs(player.x - winnable.x) + abs(player.y - winnable.y)
                    min_dist_to_winnable = min(min_dist_to_winnable, dist)
            h_cost += min_dist_to_winnable
        else:
            # If no player or winnable, this state is likely not leading to a win
            h_cost += 1000 # Large penalty

        # 2. Rule Formation Potential (proximity of rule words)
        baba_pos = None
        is_pos = None
        you_pos = None
        win_pos = None

        for y, row in enumerate(game_state.obj_map):
            for x, obj in enumerate(row):
                obj_name = self._get_object_name(obj)
                obj_type = self._get_object_type(obj)
                
                if obj_name == "baba" and obj_type == GameObjectType.Word:
                    baba_pos = (x, y)
                elif obj_name == "is" and obj_type == GameObjectType.Keyword:
                    is_pos = (x, y)
                elif obj_name == "you" and obj_type == GameObjectType.Keyword:
                    you_pos = (x, y)
                elif obj_name == "win" and obj_type == GameObjectType.Keyword:
                    win_pos = (x, y)

        # Encourage forming "BABA IS YOU" if not already active
        if "baba-is-you" not in game_state.rules:
            if baba_pos and is_pos and you_pos:
                # Calculate distance to form the rule horizontally or vertically
                dist_horizontal = float("inf")
                if baba_pos[1] == is_pos[1] and is_pos[1] == you_pos[1]:
                    dist_horizontal = max(baba_pos[0], is_pos[0], you_pos[0]) - min(baba_pos[0], is_pos[0], you_pos[0]) - 2
                
                dist_vertical = float("inf")
                if baba_pos[0] == is_pos[0] and is_pos[0] == you_pos[0]:
                    dist_vertical = max(baba_pos[1], is_pos[1], you_pos[1]) - min(baba_pos[1], is_pos[1], you_pos[1]) - 2
                
                min_dist_rule = min(dist_horizontal, dist_vertical)
                if min_dist_rule < float("inf"):
                    h_cost += min_dist_rule * 10  # Penalize distance to form rule
                else:
                    h_cost += 500 # Large penalty if not aligned
            else:
                h_cost += 1000  # Even larger penalty if key words are missing

        # Encourage forming a win condition if not already active
        win_rule_active = any("win" in rule for rule in game_state.rules)
        if not win_rule_active:
            # Look for potential win rules (any object + IS + WIN)
            for y, row in enumerate(game_state.obj_map):
                for x, obj in enumerate(row):
                    obj_name = self._get_object_name(obj)
                    obj_type = self._get_object_type(obj)
                    
                    if obj_name and obj_type == GameObjectType.Word and obj_name != "baba":
                        obj_pos = (x, y)
                        if is_pos and win_pos:
                            dist_horizontal = float("inf")
                            if obj_pos[1] == is_pos[1] and is_pos[1] == win_pos[1]:
                                dist_horizontal = max(obj_pos[0], is_pos[0], win_pos[0]) - min(obj_pos[0], is_pos[0], win_pos[0]) - 2
                            
                            dist_vertical = float("inf")
                            if obj_pos[0] == is_pos[0] and is_pos[0] == win_pos[0]:
                                dist_vertical = max(obj_pos[1], is_pos[1], win_pos[1]) - min(obj_pos[1], is_pos[1], win_pos[1]) - 2
                            
                            min_dist_win_rule = min(dist_horizontal, dist_vertical)
                            if min_dist_win_rule < float("inf"):
                                h_cost += min_dist_win_rule * 5 # Penalize distance to form win rule
                            else:
                                h_cost += 200 # Large penalty if not aligned

        # 3. Accessibility Check (simplified: check if player can reach key areas)
        if game_state.players:
            player = game_state.players[0]  # Assume single player for simplicity
            # Check if player is trapped (surrounded by non-pushable, non-passable objects)
            adjacent_positions = [
                (player.x - 1, player.y), (player.x + 1, player.y),
                (player.x, player.y - 1), (player.x, player.y + 1)
            ]
            blocked_count = 0
            for adj_x, adj_y in adjacent_positions:
                if 0 <= adj_x < len(game_state.obj_map[0]) and 0 <= adj_y < len(game_state.obj_map):
                    adj_obj = game_state.obj_map[adj_y][adj_x]
                    adj_obj_name = self._get_object_name(adj_obj)
                    if adj_obj and hasattr(adj_obj, 'is_movable') and hasattr(adj_obj, 'is_stopped'):
                        if not adj_obj.is_movable and adj_obj.is_stopped:
                            blocked_count += 1
                    elif adj_obj_name:  # If it's a string, assume it's blocking
                        blocked_count += 1
                else:
                    blocked_count += 1  # Out of bounds is also blocked
            
            if blocked_count >= 3:  # Player is mostly trapped
                h_cost += 200  # Large penalty for being trapped

        # 4. Penalty for detrimental rules (e.g., player is stop, wall is you)
        for rule in game_state.rules:
            if "you-is-stop" in rule: # If player becomes stop, it's bad
                h_cost += 1000
            if "wall-is-you" in rule: # If wall becomes you, it's bad
                h_cost += 1000
            if "baba-is-kill" in rule: # If baba becomes kill, it's bad
                h_cost += 1000

        # 5. Encourage clearing path if winnables and there are walls blocking
        if "wall-is-stop" in game_state.rules and game_state.winnables:
            for player in game_state.players:
                for winnable in game_state.winnables:
                    # Simple line-of-sight check (horizontal or vertical)
                    if player.x == winnable.x:  # Same column
                        for y in range(min(player.y, winnable.y) + 1, max(player.y, winnable.y)):
                            obj = game_state.obj_map[y][player.x]
                            obj_name = self._get_object_name(obj)
                            if obj_name == "wall":
                                h_cost += 50  # Penalty for walls blocking the path
                    elif player.y == winnable.y:  # Same row
                        for x in range(min(player.x, winnable.x) + 1, max(player.x, winnable.x)):
                            obj = game_state.obj_map[player.y][x]
                            obj_name = self._get_object_name(obj)
                            if obj_name == "wall":
                                h_cost += 50  # Penalty for walls blocking the path

        return max(0, h_cost)  # Ensure non-negative heuristic

    def _identify_subgoals(self, game_state: GameState):
        """
        Identify key sub-goals for the current game state.
        This is a simplified implementation that focuses on rule formation.
        """
        subgoals = []

        # Sub-goal 1: Ensure "BABA IS YOU" is active
        if "baba-is-you" not in game_state.rules:
            subgoals.append("form_baba_is_you")

        # Sub-goal 2: Ensure a win condition is active
        win_rule_active = any("win" in rule for rule in game_state.rules)
        if not win_rule_active:
            subgoals.append("form_win_condition")

        # Sub-goal 3: Clear path to winnable (if walls are blocking)
        if "wall-is-stop" in game_state.rules and game_state.players and game_state.winnables:
            # Check if walls are between player and winnable
            for player in game_state.players:
                for winnable in game_state.winnables:
                    if player.x == winnable.x or player.y == winnable.y:
                        # There might be walls in the direct path
                        subgoals.append("clear_path_to_winnable")
                        break

        return subgoals

    def search(self, initial_game_state: GameState, iterations: int) -> list[Direction] | None:
        # Identify sub-goals for this level
        subgoals = self._identify_subgoals(initial_game_state)

        # Priority queue for A* search: (f_cost, g_cost, hashable_state, path)
        pq = []
        heapq.heappush(pq, (0 + self._heuristic(initial_game_state), 0, self._get_hashable_state(initial_game_state), []))

        # Dictionary to store the g_cost of the shortest path found to each state
        g_costs = {self._get_hashable_state(initial_game_state): 0}

        # Dictionary to store the actual GameState objects for reconstruction
        state_map = {self._get_hashable_state(initial_game_state): initial_game_state}

        current_iterations = 0

        # Significantly increase the iteration limit for completeness
        max_iterations = 1000000  # Increased for 100% accuracy priority

        while pq and current_iterations < max_iterations:
            f_cost, g_cost, current_hashable_state, path = heapq.heappop(pq)
            current_game_state = state_map[current_hashable_state]

            # Check for win condition
            if check_win(current_game_state):
                return path

            # Explore possible actions (movements)
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
                next_game_state = copy.deepcopy(current_game_state)
                advance_game_state(action, next_game_state)
                next_hashable_state = self._get_hashable_state(next_game_state)

                new_g_cost = g_cost + 1  # Each action has a cost of 1

                if next_hashable_state not in g_costs or new_g_cost < g_costs[next_hashable_state]:
                    g_costs[next_hashable_state] = new_g_cost
                    state_map[next_hashable_state] = next_game_state
                    h_cost = self._heuristic(next_game_state)
                    heapq.heappush(pq, (new_g_cost + h_cost, new_g_cost, next_hashable_state, path + [action]))
            
            current_iterations += 1

        return None  # No solution found within the iteration limit



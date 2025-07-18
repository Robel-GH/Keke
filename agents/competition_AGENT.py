from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, GameObjectType
from typing import List, Dict, Set, Optional, Tuple, NamedTuple
from collections import deque
import heapq
import time
import json
import os
import hashlib
import math
import shutil

class SearchNode(NamedTuple):
    """Optimized search node for priority queue"""
    priority: float
    depth: int
    g_score: int
    state_hash: str
    state: GameState
    path: List[Direction]
    heuristic: float
    
    def __lt__(self, other):
        return (self.priority, self.depth, self.g_score) < (other.priority, other.depth, other.g_score)

class COMPETITIONAgent(BaseAgent):
    def __init__(self):
        # Core moves (removed Wait as it's rarely useful)
        self.moves = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
        
        # Time management
        self.max_time = 0.95
        self.strategy_time_split = [0.6, 0.25, 0.1]
        
        # Caching and learning
        self.state_cache = {}
        self.pattern_cache = {}
        self.deadlock_cache = set()
        self.success_cache = {}
        self.path_cache = {}
        self.path_cache_size = 1000
        
        # Heuristic weights
        self.weights = {
            'distance': 2.0,
            'rule_formation': 3.0,
            'push_alignment': 1.5,
            'depth_penalty': 0.1,
            'stop_penalty': 5.0,
            'kill_penalty': 10.0,
            'deadlock_penalty': 1000.0
        }
        
        # Load learned patterns
        self.load_learned_patterns()
    
    def load_learned_patterns(self):
        """Load previously learned successful patterns"""
        pattern_file = 'champion_patterns.json'
        if os.path.exists(pattern_file):
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    self.success_cache = data.get('success', {})
                    self.deadlock_cache = set(data.get('deadlocks', []))
                    self.pattern_cache = data.get('patterns', {})
            except Exception as e:
                print(f"Error loading patterns: {e}")
                pass
    
    def save_learned_patterns(self):
        """Save learned patterns for future use with better error handling"""
        try:
            pattern_file = 'champion_patterns.json'
            backup_file = 'champion_patterns.json.bak'
            
            data = {
                'success': self.success_cache,
                'deadlocks': list(self.deadlock_cache),
                'patterns': self.pattern_cache
            }
            
            # Remove backup file if it exists
            if os.path.exists(backup_file):
                os.remove(backup_file)
            
            # Create backup if original exists
            if os.path.exists(pattern_file):
                shutil.copy2(pattern_file, backup_file)
            
            # Write new data
            with open(pattern_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    def _get_cached_solution(self, state_hash: str) -> Optional[List[Direction]]:
        """Get cached solution if available"""
        return self.success_cache.get(state_hash)
    
    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """Main search with adaptive multi-strategy approach"""
        start_time = time.time()
        self.state_cache.clear()
        self.path_cache.clear()
        
        try:
            if check_win(initial_state):
                return []
            
            state_hash = self.get_state_hash(initial_state)
            
            # Check cache first
            cached_solution = self._get_cached_solution(state_hash)
            if cached_solution:
                return cached_solution
            
            # Check for trivial adjacent win
            trivial_solution = self.check_trivial_solution(initial_state)
            if trivial_solution:
                return trivial_solution
            
            # Strategy 1: Enhanced A*
            solution = self.enhanced_astar_search(initial_state, iterations, start_time)
            if solution:
                self.success_cache[state_hash] = solution
                self.save_learned_patterns()
                return solution
            
            # Strategy 2: Intelligent DFS
            elapsed = time.time() - start_time
            if elapsed < self.max_time * sum(self.strategy_time_split[:2]):
                solution = self.intelligent_dfs(initial_state, min(iterations, 35), start_time)
                if solution:
                    self.success_cache[state_hash] = solution
                    self.save_learned_patterns()
                    return solution
            
            # Strategy 3: Optimized BFS
            elapsed = time.time() - start_time
            if elapsed < self.max_time * 0.9:
                solution = self.optimized_bfs(initial_state, min(iterations // 2, 25))
                if solution:
                    self.success_cache[state_hash] = solution
                    self.save_learned_patterns()
                    return solution
            
            return []
            
        except Exception as e:
            print(f"Critical error in search: {e}")
            return []
    
    def check_trivial_solution(self, state: GameState) -> List[Direction]:
        """Check if win is adjacent to player"""
        try:
            if state.players and state.winnables:
                for player in state.players:
                    for win in state.winnables:
                        dx = abs(player.x - win.x)
                        dy = abs(player.y - win.y)
                        if dx + dy == 1:  # Adjacent
                            if dx == 1:
                                return [Direction.Right] if player.x < win.x else [Direction.Left]
                            else:
                                return [Direction.Down] if player.y < win.y else [Direction.Up]
        except Exception:
            pass
        return []
    
    def enhanced_astar_search(self, initial_state: GameState, iterations: int, start_time: float) -> List[Direction]:
        """Optimized A* search with advanced heuristics"""
        try:
            open_list = []
            counter = 0
            
            initial_h = self.calculate_advanced_heuristic(initial_state, 0)
            if initial_h >= 1000.0:
                return []
            
            initial_node = SearchNode(
                priority=initial_h,
                depth=0,
                g_score=0,
                state_hash=self.get_state_hash(initial_state),
                state=initial_state,
                path=[],
                heuristic=initial_h
            )
            
            heapq.heappush(open_list, initial_node)
            visited = {}
            best_solution = None
            best_heuristic = float('inf')
            
            while open_list:
                if time.time() - start_time > self.max_time * self.strategy_time_split[0]:
                    break
                
                current = heapq.heappop(open_list)
                
                if check_win(current.state):
                    return current.path
                
                # Skip if we have a better path to this state
                if current.state_hash in visited and visited[current.state_hash] <= current.g_score:
                    continue
                visited[current.state_hash] = current.g_score
                
                if self.is_deadlock_state(current.state):
                    continue
                
                if current.depth >= min(iterations, 60):
                    continue
                
                moves = self.get_ordered_moves(current.state)
                for move in moves:
                    next_state = advance_game_state(move, current.state.copy())
                    if next_state is None:
                        continue
                    
                    next_hash = self.get_state_hash(next_state)
                    if next_hash in visited and visited[next_hash] <= current.g_score + 1:
                        continue
                    
                    g_score = current.g_score + 1
                    h_score = self.calculate_advanced_heuristic(next_state, g_score)
                    
                    if h_score < 1000.0:
                        priority = g_score + h_score
                        next_node = SearchNode(
                            priority=priority,
                            depth=current.depth + 1,
                            g_score=g_score,
                            state_hash=next_hash,
                            state=next_state,
                            path=current.path + [move],
                            heuristic=h_score
                        )
                        
                        heapq.heappush(open_list, next_node)
                        
                        # Track best solution found so far
                        if h_score < best_heuristic:
                            best_heuristic = h_score
                            best_solution = next_node.path
            
            return best_solution if best_solution else []
            
        except Exception as e:
            print(f"Error in A* search: {e}")
            return []
    
    def intelligent_dfs(self, initial_state: GameState, max_depth: int, start_time: float) -> List[Direction]:
        """DFS with intelligent pruning and ordering"""
        try:
            stack = [(initial_state, [], 0)]
            visited = set()
            best_solution = None
            best_heuristic = float('inf')
            
            while stack:
                if time.time() - start_time > self.max_time * (self.strategy_time_split[0] + self.strategy_time_split[1]):
                    break
                
                current_state, path, depth = stack.pop()
                
                if check_win(current_state):
                    return path
                
                state_hash = self.get_state_hash(current_state)
                if state_hash in visited or depth >= max_depth or self.is_deadlock_state(current_state):
                    continue
                visited.add(state_hash)
                
                moves = self.get_ordered_moves(current_state)
                for move in reversed(moves):
                    next_state = advance_game_state(move, current_state.copy())
                    if next_state is not None:
                        next_hash = self.get_state_hash(next_state)
                        if next_hash not in visited:
                            # Calculate heuristic for pruning
                            h = self.calculate_advanced_heuristic(next_state, depth + 1)
                            if h < best_heuristic:
                                best_heuristic = h
                                best_solution = path + [move]
                            stack.append((next_state, path + [move], depth + 1))
            
            return best_solution if best_solution else []
            
        except Exception as e:
            print(f"Error in DFS search: {e}")
            return []
    
    def optimized_bfs(self, initial_state: GameState, max_depth: int) -> List[Direction]:
        """Optimized BFS with smart pruning"""
        try:
            queue = deque([(initial_state, [])])
            visited = set()
            
            while queue:
                current_state, path = queue.popleft()
                
                if check_win(current_state):
                    return path
                
                state_hash = self.get_state_hash(current_state)
                if state_hash in visited or len(path) >= max_depth:
                    continue
                visited.add(state_hash)
                
                if self.is_simple_deadlock(current_state):
                    continue
                
                for move in self.moves:
                    next_state = advance_game_state(move, current_state.copy())
                    if next_state is not None:
                        next_hash = self.get_state_hash(next_state)
                        if next_hash not in visited:
                            queue.append((next_state, path + [move]))
            
            return []
            
        except Exception as e:
            print(f"Error in BFS search: {e}")
            return []
    
    def get_state_hash(self, state: GameState) -> str:
        """Generate efficient state hash based on object positions and rules"""
        try:
            phys_repr = tuple(sorted(
                (obj.x, obj.y, obj.name, obj.object_type.name)
                for obj in state.phys
            ))
            rule_repr = tuple(sorted(state.rules))
            state_repr = (phys_repr, rule_repr)
            return hashlib.md5(str(state_repr).encode()).hexdigest()
        except Exception:
            return str(hash(str(state)))
    
    def calculate_advanced_heuristic(self, state: GameState, depth: int) -> float:
        """Advanced heuristic combining multiple factors"""
        try:
            if check_win(state):
                return 0.0
            
            state_hash = self.get_state_hash(state)
            if state_hash in self.state_cache:
                return self.state_cache[state_hash]
            
            # Check for YOU rule
            has_you_rule = self.has_property_rule(state, "YOU")
            if not has_you_rule:
                possible_you_rules = self.get_possible_rules(state, "YOU")
                if not possible_you_rules:
                    return 1000.0
                min_cost = min(self.calculate_rule_formation_cost(state, X, "IS", "YOU") 
                             for X in possible_you_rules)
                self.state_cache[state_hash] = min_cost
                return min_cost
            
            # Check for WIN rule
            has_win_rule = self.has_property_rule(state, "WIN")
            if has_win_rule:
                if not state.players or not state.winnables:
                    return 1000.0
                
                min_dist = float('inf')
                for player in state.players:
                    for win in state.winnables:
                        dist = self.calculate_obstacle_aware_distance(
                            state, 
                            (player.x, player.y), 
                            (win.x, win.y)
                        )
                        if dist < min_dist:
                            min_dist = dist
                
                # Add penalty for dangerous objects
                danger_penalty = self.calculate_danger_penalty(state, player, state.winnables)
                total_cost = min_dist * self.weights['distance'] + danger_penalty + depth * self.weights['depth_penalty']
                
                self.state_cache[state_hash] = total_cost
                return total_cost
            
            # Need to form WIN rule
            possible_win_rules = self.get_possible_rules(state, "WIN")
            if not possible_win_rules:
                return 1000.0
            
            min_cost = 1000.0
            for Y in possible_win_rules:
                formation_cost = self.calculate_rule_formation_cost(state, Y, "IS", "WIN")
                if formation_cost >= 1000.0:
                    continue
                
                Y_objs = [obj for obj in state.phys if obj.name == Y]
                if not Y_objs or not state.players:
                    continue
                
                min_dist = float('inf')
                for player in state.players:
                    for y_obj in Y_objs:
                        dist = self.calculate_obstacle_aware_distance(
                            state,
                            (player.x, player.y),
                            (y_obj.x, y_obj.y)
                        )
                        if dist < min_dist:
                            min_dist = dist
                
                total_cost = formation_cost * self.weights['rule_formation'] + min_dist * self.weights['distance']
                min_cost = min(min_cost, total_cost)
            
            self.state_cache[state_hash] = min_cost
            return min_cost
            
        except Exception as e:
            print(f"Heuristic error: {str(e)}")
            return 1000.0
    
    def has_property_rule(self, state: GameState, property: str) -> bool:
        """Check if any object has the given property"""
        try:
            for rule in state.rules:
                parts = rule.split()
                if len(parts) >= 3 and parts[-1] == property and "IS" in parts:
                    # Check if rule is not negated
                    if "NOT" not in parts:
                        return True
            return False
        except Exception:
            return False
    
    def calculate_rule_formation_cost(self, state: GameState, A: str, B: str, C: str) -> float:
        """Efficient rule formation cost using median alignment"""
        try:
            # Get all positions for each word
            pos_A_list = [(obj.x, obj.y) for obj in state.phys 
                         if obj.name == A and obj.object_type == GameObjectType.Word]
            pos_B_list = [(obj.x, obj.y) for obj in state.phys 
                         if obj.name == B and obj.object_type == GameObjectType.Word]
            pos_C_list = [(obj.x, obj.y) for obj in state.phys 
                         if obj.name == C and obj.object_type == GameObjectType.Word]
            
            if not pos_A_list or not pos_B_list or not pos_C_list:
                return 1000.0
            
            min_cost = 1000.0
            # Try all combinations of word positions
            for pos_A in pos_A_list:
                for pos_B in pos_B_list:
                    for pos_C in pos_C_list:
                        # Horizontal alignment
                        xs = [pos_A[0], pos_B[0] - 1, pos_C[0] - 2]
                        ys = [pos_A[1], pos_B[1], pos_C[1]]
                        x_opt = sorted(xs)[1]  # Median x
                        y_opt = sorted(ys)[1]  # Median y
                        cost_h = (
                            abs(pos_A[0] - x_opt) + abs(pos_A[1] - y_opt) +
                            abs(pos_B[0] - (x_opt+1)) + abs(pos_B[1] - y_opt) +
                            abs(pos_C[0] - (x_opt+2)) + abs(pos_C[1] - y_opt)
                        )
                        
                        # Vertical alignment
                        xs = [pos_A[0], pos_B[0], pos_C[0]]
                        ys = [pos_A[1], pos_B[1] - 1, pos_C[1] - 2]
                        x_opt_v = sorted(xs)[1]
                        y_opt_v = sorted(ys)[1]
                        cost_v = (
                            abs(pos_A[0] - x_opt_v) + abs(pos_A[1] - y_opt_v) +
                            abs(pos_B[0] - x_opt_v) + abs(pos_B[1] - (y_opt_v+1)) +
                            abs(pos_C[0] - x_opt_v) + abs(pos_C[1] - (y_opt_v+2))
                        )
                        
                        min_cost = min(min_cost, cost_h, cost_v)
            
            return min_cost
            
        except Exception:
            return 1000.0
    
    def calculate_obstacle_aware_distance(self, state: GameState, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """BFS pathfinding considering STOP objects"""
        try:
            # Check cache first
            cache_key = (start, end, self.get_state_hash(state))
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]
            
            queue = deque([(start, 0)])
            visited = {start}
            grid = state.obj_map
            
            while queue:
                (x, y), dist = queue.popleft()
                if (x, y) == end:
                    # Update cache
                    if len(self.path_cache) >= self.path_cache_size:
                        self.path_cache.popitem()
                    self.path_cache[cache_key] = dist
                    return dist
                    
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        # Skip if cell contains STOP object
                        if not any(obj.is_stopped for obj in grid[ny][nx]):
                            if (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append(((nx, ny), dist + 1))
            
            # Fallback to Manhattan distance
            dist = abs(start[0]-end[0]) + abs(start[1]-end[1])
            if len(self.path_cache) >= self.path_cache_size:
                self.path_cache.popitem()
            self.path_cache[cache_key] = dist
            return dist
            
        except Exception:
            # Fallback to Manhattan distance
            return abs(start[0]-end[0]) + abs(start[1]-end[1])
    
    def calculate_danger_penalty(self, state: GameState, player, win_objects) -> float:
        """Calculate penalty for dangerous objects near the path"""
        try:
            penalty = 0.0
            if not win_objects:
                return penalty
                
            # Find nearest win object
            win = min(win_objects, key=lambda w: abs(player.x - w.x) + abs(player.y - w.y))
            
            # Check for KILL objects near the path
            for obj in state.phys:
                if obj.is_kill:
                    # Simple line-of-sight check
                    if self.is_near_line((player.x, player.y), (win.x, win.y), (obj.x, obj.y), 2):
                        penalty += self.weights['kill_penalty']
            
            # Check for STOP objects blocking path
            for obj in state.phys:
                if obj.is_stopped and not obj.is_pushable:
                    if self.is_between((player.x, player.y), (win.x, win.y), (obj.x, obj.y)):
                        penalty += self.weights['stop_penalty']
            
            return penalty
            
        except Exception:
            return 0.0
    
    def is_near_line(self, p1, p2, test_point, threshold=2) -> bool:
        """Check if point is near the line between p1 and p2"""
        try:
            x0, y0 = test_point
            x1, y1 = p1
            x2, y2 = p2
            
            # Handle case where p1 and p2 are the same
            if x1 == x2 and y1 == y2:
                dist = math.sqrt((x0-x1)**2 + (y0-y1)**2)
                return dist <= threshold
            
            # Calculate distance from point to line
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = math.sqrt((y2-y1)**2 + (x2-x1)**2)
            dist = numerator / denominator
            
            return dist <= threshold
        except Exception:
            return False
    
    def is_between(self, p1, p2, test_point) -> bool:
        """Check if test_point is between p1 and p2"""
        try:
            x0, y0 = test_point
            x1, y1 = p1
            x2, y2 = p2
            
            # Check if test_point is in the bounding box
            if not (min(x1, x2) <= x0 <= max(x1, x2)) or not (min(y1, y2) <= y0 <= max(y1, y2)):
                return False
                
            # Check alignment (cross product should be near zero)
            cross = (y0 - y1) * (x2 - x1) - (x0 - x1) * (y2 - y1)
            return abs(cross) < 1
        except Exception:
            return False
    
    def get_possible_rules(self, state: GameState, property: str) -> List[str]:
        """Identify objects that can form rules with given property"""
        try:
            possible = set()
            for obj in state.phys:
                if obj.object_type == GameObjectType.Word:
                    if all(self.text_exists(state, word) for word in [obj.name, "IS", property]):
                        possible.add(obj.name)
            return list(possible)
        except Exception:
            return []
    
    def text_exists(self, state: GameState, word: str) -> bool:
        """Check if text block exists"""
        try:
            return any(obj.name == word and obj.object_type == GameObjectType.Word 
                      for obj in state.phys)
        except Exception:
            return False
    
    def get_ordered_moves(self, state: GameState) -> List[Direction]:
        """Order moves by heuristic promise"""
        try:
            move_scores = []
            
            for move in self.moves:
                next_state = advance_game_state(move, state.copy())
                if next_state is not None:
                    score = self.calculate_advanced_heuristic(next_state, 0)
                    move_scores.append((score, move))
            
            move_scores.sort(key=lambda x: x[0])
            return [move for _, move in move_scores]
        except Exception:
            return self.moves
    
    def is_deadlock_state(self, state: GameState) -> bool:
        """Advanced deadlock detection with multiple checks"""
        try:
            state_hash = self.get_state_hash(state)
            if state_hash in self.deadlock_cache:
                return True
            
            # 1. Check for trapped pushables
            if state.pushables:
                for push in state.pushables:
                    if self.is_trapped(state, push.x, push.y):
                        self.deadlock_cache.add(state_hash)
                        return True
            
            # 2. Check for essential word destruction
            essential_words = {"IS", "YOU", "WIN"}
            existing_words = {obj.name for obj in state.phys 
                             if obj.object_type == GameObjectType.Word}
            if not all(word in existing_words for word in essential_words):
                self.deadlock_cache.add(state_hash)
                return True
                
            # 3. Check if YOU rule can be reformed
            if not state.players and not self.can_reform_you_rule(state):
                self.deadlock_cache.add(state_hash)
                return True
                
            return False
        except Exception:
            return False
    
    def is_trapped(self, state: GameState, x: int, y: int) -> bool:
        """Check if object at (x,y) is completely blocked"""
        try:
            blocked = 0
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < len(state.obj_map[0]) and 0 <= ny < len(state.obj_map)):
                    blocked += 1
                    continue
                cell = state.obj_map[ny][nx]
                # If any object in the cell is not pushable and is stop, then blocked
                if any(not obj.is_pushable and obj.is_stopped for obj in cell):
                    blocked += 1
            return blocked >= 3
        except Exception:
            return False
    
    def can_reform_you_rule(self, state: GameState) -> bool:
        """Check if a YOU rule can potentially be reformed"""
        try:
            return any(obj.name == "IS" for obj in state.phys) and \
                   any(obj.name == "YOU" for obj in state.phys) and \
                   any(obj.object_type == GameObjectType.WordNoun for obj in state.phys)
        except Exception:
            return False
    
    def is_simple_deadlock(self, state: GameState) -> bool:
        """Quick deadlock check for BFS"""
        try:
            if not state.pushables:
                return False
            
            for push in state.pushables:
                blocked = 0
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = push.x + dx, push.y + dy
                    if not (0 <= nx < len(state.obj_map[0]) and 0 <= ny < len(state.obj_map)):
                        blocked += 1
                        continue
                    cell = state.obj_map[ny][nx]
                    if any(not obj.is_pushable and obj.is_stopped for obj in cell):
                        blocked += 1
                if blocked >= 3:
                    return True
            
            return False
        except Exception:
            return False
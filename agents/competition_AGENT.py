"""
Competition-optimized Learning Hybrid A* Agent
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Dict, Set, NamedTuple
from collections import deque
import heapq
import json
import os
import time

class QueueItem(NamedTuple):
    """Safe comparison wrapper for priority queue"""
    priority: float
    depth: int
    state_hash: str
    state: GameState
    path: List[Direction]
    
    def __lt__(self, other):
        return (self.priority, self.depth) < (other.priority, other.depth)

class CompetitionAgent(BaseAgent):
    def __init__(self):
        # Core configuration
        self.moves = [Direction.Right, Direction.Up, Direction.Left, Direction.Down, Direction.Wait]
        self.max_time = 0.95  # 95% of available time
        
        # Learning components
        self.pattern_memory = {}  # Learned state patterns
        self.success_patterns = {}  # Known successful solutions
        self.level_patterns = {}  # Level-specific patterns
        
        # Competition optimization
        self.state_cache = {}  # Current session cache
        self.deadlock_patterns = set()  # Known deadlock states
        self.performance_stats = {}  # Track solution performance
        
        # Load learned patterns
        self.load_patterns()

    def load_patterns(self):
        """Load previously learned patterns"""
        pattern_file = f'competition_patterns_{os.getenv("USER", "default")}.json'
        if os.path.exists(pattern_file):
            with open(pattern_file, 'r') as f:
                data = json.load(f)
                self.pattern_memory = data.get('patterns', {})
                self.success_patterns = data.get('success', {})
                self.level_patterns = data.get('levels', {})
                self.deadlock_patterns = set(data.get('deadlocks', []))

    def save_patterns(self):
        """Save learned patterns"""
        pattern_file = f'competition_patterns_{os.getenv("USER", "default")}.json'
        data = {
            'patterns': self.pattern_memory,
            'success': self.success_patterns,
            'levels': self.level_patterns,
            'deadlocks': list(self.deadlock_patterns)
        }
        with open(pattern_file, 'w') as f:
            json.dump(data, f)

    def detect_deadlock(self, state: GameState) -> bool:
        """Quick deadlock detection"""
        state_hash = str(state)
        if state_hash in self.deadlock_patterns:
            return True
            
        # Check for blocked push objects
        if state.pushables:
            for push in state.pushables:
                blocked_count = 0
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    x, y = push.x + dx, push.y + dy
                    # Check if position is blocked
                    for obj in state.pushables:
                        if obj.x == x and obj.y == y:
                            blocked_count += 1
                            break
                            
                if blocked_count >= 3:  # Object is trapped
                    self.deadlock_patterns.add(state_hash)
                    return True
                    
        return False

    def calculate_heuristic(self, state: GameState, depth: int) -> float:
        """Competition-optimized heuristic"""
        if check_win(state):
            return 0.0
            
        state_hash = str(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]

        score = 0.0
        
        # Multi-objective scoring
        if state.players and state.winnables:
            # 1. Distance-based scoring
            min_dist = float('inf')
            for player in state.players:
                for win in state.winnables:
                    dist = abs(player.x - win.x) + abs(player.y - win.y)
                    min_dist = min(min_dist, dist)
            score += min_dist * 2.0

            # 2. Push object scoring
            if state.pushables:
                push_score = 0.0
                for push in state.pushables:
                    for win in state.winnables:
                        push_dist = abs(push.x - win.x) + abs(push.y - win.y)
                        if push_dist < 3:
                            push_score -= 2.0  # Reward close pushables
                score += push_score

            # 3. Pattern matching bonus
            pattern_key = self.get_pattern_key(state)
            if pattern_key in self.pattern_memory:
                score -= 5.0  # Heavy reward for known patterns

            # 4. Depth penalty
            score += depth * 0.1  # Small penalty for depth

        self.state_cache[state_hash] = score
        return score

    def get_pattern_key(self, state: GameState) -> str:
        """Generate pattern key for state"""
        # Create simplified state representation for pattern matching
        key_parts = []
        if state.players:
            key_parts.append(f"p{state.players[0].x},{state.players[0].y}")
        if state.winnables:
            key_parts.append(f"w{state.winnables[0].x},{state.winnables[0].y}")
        if state.pushables:
            pushables = sorted(f"b{p.x},{p.y}" for p in state.pushables)
            key_parts.extend(pushables)
        return "|".join(key_parts)

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """Competition-optimized search"""
        start_time = time.time()
        self.state_cache.clear()
        
        # Check for known solution
        state_hash = str(initial_state)
        if state_hash in self.success_patterns:
            return self.success_patterns[state_hash]

        # Initialize search
        initial_priority = self.calculate_heuristic(initial_state, 0)
        queue = [QueueItem(
            priority=initial_priority,
            depth=0,
            state_hash=state_hash,
            state=initial_state,
            path=[]
        )]
        heapq.heapify(queue)
        
        visited = set()
        best_solution = []
        best_length = float('inf')
        
        steps = 0
        while queue and steps < iterations:
            # Time check
            if time.time() - start_time > self.max_time:
                break
                
            current = heapq.heappop(queue)
            steps += 1

            # Win check
            if check_win(current.state):
                if len(current.path) < best_length:
                    best_solution = current.path
                    best_length = len(current.path)
                    # Store successful pattern
                    self.success_patterns[state_hash] = current.path
                continue

            # Skip visited or deadlocked states
            if current.state_hash in visited or self.detect_deadlock(current.state):
                continue
            visited.add(current.state_hash)

            # Explore moves if within limits
            if current.depth < iterations:
                for move in self.moves:
                    next_state = advance_game_state(move, current.state.copy())
                    if next_state is None:
                        continue
                        
                    next_hash = str(next_state)
                    if next_hash in visited:
                        continue

                    # Calculate priority using enhanced heuristic
                    next_priority = self.calculate_heuristic(next_state, current.depth + 1)
                    
                    # Create new queue item
                    next_item = QueueItem(
                        priority=next_priority,
                        depth=current.depth + 1,
                        state_hash=next_hash,
                        state=next_state,
                        path=current.path + [move]
                    )
                    
                    heapq.heappush(queue, next_item)

        # Learn from this execution
        if best_solution:
            self.pattern_memory[self.get_pattern_key(initial_state)] = best_solution
            self.save_patterns()

        return best_solution if best_solution else []
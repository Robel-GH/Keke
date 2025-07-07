"""
Fixed Hybrid A* implementation for complex game scenarios
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Set, Tuple, NamedTuple
from collections import deque
import heapq

class QueueItem(NamedTuple):
    """Wrapper class to make items comparable in priority queue"""
    priority: float
    depth: int
    state_hash: str
    state: GameState
    path: List[Direction]
    
    def __lt__(self, other):
        # Primary sort by priority, secondary by depth
        return (self.priority, self.depth) < (other.priority, other.depth)

class HYBRID_ASTARAgent(BaseAgent):
    def __init__(self):
        self.moves = [
            Direction.Right, 
            Direction.Up, 
            Direction.Left, 
            Direction.Down, 
            Direction.Wait
        ]
        self.visited = set()
        self.state_cache = {}

    def calculate_heuristic(self, state: GameState) -> float:
        """Calculate heuristic value for state"""
        if check_win(state):
            return 0.0
            
        # Use cache to avoid recalculation
        state_str = str(state)
        if state_str in self.state_cache:
            return self.state_cache[state_str]

        score = float('inf')
        
        # Calculate minimum distance to win condition
        if state.players and state.winnables:
            min_dist = float('inf')
            for player in state.players:
                for win in state.winnables:
                    dist = abs(player.x - win.x) + abs(player.y - win.y)
                    min_dist = min(min_dist, dist)
            score = min_dist

            # Add penalty for pushables in the way
            if state.pushables:
                for push in state.pushables:
                    for player in state.players:
                        for win in state.winnables:
                            if (min(player.x, win.x) <= push.x <= max(player.x, win.x) and
                                min(player.y, win.y) <= push.y <= max(player.y, win.y)):
                                score += 0.5

        self.state_cache[state_str] = score
        return score

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """Enhanced search with proper object comparison"""
        self.visited.clear()
        self.state_cache.clear()
        
        # Initialize queue with wrapped QueueItem
        initial_priority = self.calculate_heuristic(initial_state)
        initial_item = QueueItem(
            priority=initial_priority,
            depth=0,
            state_hash=str(initial_state),
            state=initial_state,
            path=[]
        )
        
        queue = [initial_item]
        heapq.heapify(queue)
        
        best_path = []
        steps = 0
        
        while queue and steps < iterations:
            current = heapq.heappop(queue)
            steps += 1

            # Check win condition
            if check_win(current.state):
                if not best_path or len(current.path) < len(best_path):
                    best_path = current.path
                continue

            # Skip if state already visited
            if current.state_hash in self.visited:
                continue
            self.visited.add(current.state_hash)

            # Explore moves if within depth limit
            if current.depth < iterations:
                for move in self.moves:
                    next_state = advance_game_state(move, current.state.copy())
                    if next_state is None:
                        continue
                        
                    next_hash = str(next_state)
                    if next_hash in self.visited:
                        continue

                    # Calculate priority for next state
                    next_priority = self.calculate_heuristic(next_state)
                    
                    # Create new queue item
                    next_item = QueueItem(
                        priority=next_priority,
                        depth=current.depth + 1,
                        state_hash=next_hash,
                        state=next_state,
                        path=current.path + [move]
                    )
                    
                    heapq.heappush(queue, next_item)

        return best_path if best_path else []
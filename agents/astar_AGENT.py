"""
Implementation of A* agent for KekeAI.
Uses A* search algorithm with heuristic functions to find optimal solutions.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win
from typing import List, Set, Tuple, NamedTuple
import heapq
from tqdm import trange


class QueueItem(NamedTuple):
    """Helper class to make states comparable in priority queue"""
    f_score: float
    g_score: float
    moves: int
    state: GameState
    path: List[Direction]

    def __lt__(self, other):
        # Compare only by f_score and moves (not by GameState)
        return (self.f_score, self.moves) < (other.f_score, other.moves)


class ASTARAgent(BaseAgent):
    """
    A* Search implementation with heuristic function.
    """

    def __init__(self):
        self.moves = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions"""
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def calculate_heuristic(self, state: GameState) -> float:
        """
        Calculate heuristic value for a state.
        Lower values are better.
        """
        if check_win(state):
            return 0.0
            
        # Find player positions
        player_positions = [(p.x, p.y) for p in state.players]
        if not player_positions:
            return float('inf')
            
        # Find win positions
        win_positions = [(w.x, w.y) for w in state.winnables]
        if not win_positions:
            return float('inf')
            
        # Calculate minimum distance to any win position
        min_distance = float('inf')
        for player_pos in player_positions:
            for win_pos in win_positions:
                dist = self.manhattan_distance(player_pos, win_pos)
                min_distance = min(min_distance, dist)
                
        return min_distance

    def get_state_hash(self, state: GameState) -> str:
        """Create a unique hash for the state"""
        # Include object positions and rules
        obj_positions = [f"{obj.name}:{obj.x},{obj.y}" for obj in state.objects]
        rules = sorted(state.rules)
        return f"objs:[{';'.join(sorted(obj_positions))}]rules:[{';'.join(rules)}]"

    def search(self, initial_state: GameState, iterations: int = 50) -> List[Direction]:
        """
        A* search implementation.
        """
        # Initialize search
        start_h = self.calculate_heuristic(initial_state)
        start_item = QueueItem(start_h, 0.0, 0, initial_state, [])
        queue = [start_item]
        heapq.heapify(queue)
        
        # Track visited states
        visited = set()
        moves_made = 0
        
        # For tracking best partial solution
        best_h = float('inf')
        best_path = []
        
        while queue and moves_made < iterations:
            # Get most promising state
            current = heapq.heappop(queue)
            current_state = current.state
            current_path = current.path
            
            # Check win condition
            if check_win(current_state):
                return current_path
            
            # Update best path if this state looks promising
            if current.f_score < best_h:
                best_h = current.f_score
                best_path = current_path
            
            # Check if state was visited
            state_hash = self.get_state_hash(current_state)
            if state_hash in visited:
                continue
                
            # Mark as visited and increment counter
            visited.add(state_hash)
            moves_made += 1
            
            # Try each possible move
            for move in self.moves:
                # Generate new state
                next_state = advance_game_state(move, current_state.copy())
                if next_state is None:
                    continue
                    
                # Skip if already visited
                next_hash = self.get_state_hash(next_state)
                if next_hash in visited:
                    continue
                
                # Calculate costs
                g_score = current.g_score + 1
                h_score = self.calculate_heuristic(next_state)
                f_score = g_score + h_score
                
                # Create queue item and add to queue
                next_item = QueueItem(
                    f_score=f_score,
                    g_score=g_score,
                    moves=moves_made,
                    state=next_state,
                    path=current_path + [move]
                )
                heapq.heappush(queue, next_item)
        
        # If no solution found, return best partial solution
        return best_path
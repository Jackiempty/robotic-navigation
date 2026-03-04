# your_implementation/a_star_implementation.py
import cv2
import numpy as np

from path_planning import *
from path_planning.a_star_planner import AStarPlanner


class AStarImplementation(AStarPlanner):
    # TODO: implement your own version of preloop, step and postloop
    def preloop(self):
        # 1. Initialize data structure
        self.open_set: set[PathNode] = {self.start_node}  # nodes to be explored
        self.closed_set: set[PathNode] = set()            # explored nodes
        self.visited_nodes: set[PathNode] = {self.start_node} # will be in drawing
        
        # 2. Initialize value of g and h
        self.g: dict[PathNode, float] = {self.start_node: 0}
        self.h: dict[PathNode, float] = {
            self.start_node: calculate_node_distance(self.start_node, self.goal_node)
        }

    def step(self):
        # If open_set is empty, meaning the path cannot be found, the loop ends
        if not self.open_set:
            self.is_done.set()
            return

        # Look up the node with smallest g + h in open_set
        current_node = min(
            self.open_set, 
            key=lambda node: (
                self.g.get(node, float('inf')) + self.h.get(node, float('inf')), 
                -self.g.get(node, 0) # Add minus to let node with greater g line backer
            )
        )

        # Check whether arriving the goal (smaller than threshold)
        dist_to_goal = calculate_node_distance(current_node, self.goal_node)
        if dist_to_goal <= self.goal_threshold:
            # Put on the parent of goal to current node so postloop can trace
            self.goal_node.parent = current_node
            self.visited_nodes.add(self.goal_node)
            self.is_done.set() # trigger end condition
            return

        # Remove current node from open_set and add to closed_set & visited_node
        self.open_set.remove(current_node)
        self.closed_set.add(current_node)

        self.visited_nodes.add(current_node)

        # Expand neighbor nodes
        for neighbor in self.get_neighbor_nodes(current_node):
            if neighbor in self.closed_set:
                continue # If explored, skip

            # Calculate tentative_g：The distance from start_node to current_node，then to neighbor
            move_cost = calculate_node_distance(current_node, neighbor)
            tentative_g = self.g[current_node] + move_cost

            # If it is the first time meeting this neighbor, or a shorter path is found to get to this neighbor
            if neighbor not in self.open_set or tentative_g < self.g.get(neighbor, float('inf')):
                # Update status
                neighbor.parent = current_node # Record the parent for easier trace
                self.g[neighbor] = tentative_g
                self.h[neighbor] = calculate_node_distance(neighbor, self.goal_node)
                
                self.open_set.add(neighbor)
                # self.visited_nodes.add(neighbor)

    def postloop(self):
        if self.goal_node.parent is None:
            print("Warning: Path not found!")
            return [], self.visited_nodes
            
        path = collect_path(self.goal_node)
        return path, self.visited_nodes

# your_implementation/rrt_star_implementation.py
import cv2
import numpy as np

from path_planning import *
from path_planning.rrt_star_planner import RRTStarPlanner


class RRTStarImplementation(RRTStarPlanner):
    # TODO: implement your own version of preloop, step and postloop
    def preloop(self):
        # Initialize tree, and put start node into the tree
        self.tree: list[PathNode] = [self.start_node]
        self.start_node.cost = 0

    def step(self):
        # Sample: Randomly sample a new node
        random_node = self.sample_random_node()

        # Nearest: Find the node in the tree which is nearest to the random node
        nearest_node = min(
            self.tree, 
            key=lambda n: calculate_node_distance(n, random_node)
        )

        # Steer: Walk a distance of step_size from nearest_node to random_node
        dist_to_random = calculate_node_distance(nearest_node, random_node)
        
        if dist_to_random <= self.step_size:
            # If the random_node is already under the threshold of step_size, step on it
            new_x = random_node.coordinates.x
            new_y = random_node.coordinates.y
        else:
            # Otherwise calculate the coordinate after stepping forward for step_size by fraction
            dx = random_node.coordinates.x - nearest_node.coordinates.x
            dy = random_node.coordinates.y - nearest_node.coordinates.y
            new_x = nearest_node.coordinates.x + (dx / dist_to_random) * self.step_size
            new_y = nearest_node.coordinates.y + (dy / dist_to_random) * self.step_size

        new_node = PathNode(coordinates=PixelCoordinates(new_x, new_y))

        # Make sure the new node doesn't be out of the map
        if not check_inside_map(self.occupancy_map, new_node):
            return

        # Check whether there is obstacle between nearest_node and new_node
        if not check_collision_free(self.occupancy_map, nearest_node, new_node):
            return

        # --- RRT* exclusive optimization ---

        # Find a set of nearest nodes within the range of search_radius
        near_nodes = [
            n for n in self.tree 
            if calculate_node_distance(n, new_node) <= self.search_radius
        ]

        # Choose Parent: Pick the node with lowest cost as parent node from nearest nodes 
        min_cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)
        best_parent = nearest_node

        for near_node in near_nodes:
            # Calculate the total cost from near_node
            potential_cost = near_node.cost + calculate_node_distance(near_node, new_node)
            if potential_cost < min_cost and check_collision_free(self.occupancy_map, near_node, new_node):
                min_cost = potential_cost
                best_parent = near_node

        # Finally put the new node onto the tree
        new_node.parent = best_parent
        new_node.cost = min_cost
        self.tree.append(new_node)

        # Rewire: Check whether it is cheaper if the new node is set as parent node to nearest node
        for near_node in near_nodes:
            rewire_cost = new_node.cost + calculate_node_distance(new_node, near_node)
            if rewire_cost < near_node.cost and check_collision_free(self.occupancy_map, new_node, near_node):
                # Rewire
                near_node.parent = new_node
                near_node.cost = rewire_cost

        # Check arriving yet
        dist_to_goal = calculate_node_distance(new_node, self.goal_node)
        if dist_to_goal <= self.goal_threshold:
            # Make sure the last step to goal is not blocked
            if check_collision_free(self.occupancy_map, new_node, self.goal_node):
                self.goal_node.parent = new_node
                self.goal_node.cost = new_node.cost + dist_to_goal
                self.tree.append(self.goal_node)
                self.is_done.set() # End after found a path

    def postloop(self):
        if self.goal_node.parent is None:
            print("Warning: RRT* failed to find a path within the iteration limit!")
            return [], set(self.tree)
            
        path = collect_path(self.goal_node)
        return path, set(self.tree)

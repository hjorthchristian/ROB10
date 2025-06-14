#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from stack_optimization_interfaces.srv import StackOptimizer
from ortools.sat.python import cp_model
import numpy as np
from tf2_ros import TransformBroadcaster
from tf2_msgs.msg import TFMessage
from transforms3d.euler import euler2quat
import matplotlib
# Force matplotlib to use a specific backend that works with ROS
matplotlib.use('TkAgg')  # This should work on most systems
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from stack_optimization_interfaces.msg import BoxState, BoxInfo
from std_msgs.msg import Header
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2mat

class BoxStackOptimizer:
    def __init__(self, pallet_x=80, pallet_y=60, max_height=100, margin=10):
        """
        Initialize the optimizer with pallet dimensions and safety margin.
        
        Note: In this implementation, we assume:
        - x corresponds to width 
        - y corresponds to length
        - This is opposite of the original implementation
        
        Args:
            pallet_x: Width of the pallet in optimizer units
            pallet_y: Length of the pallet in optimizer units
            max_height: Maximum stacking height
            margin: Safety margin between boxes
        """
        self.pallet_x = pallet_x  # Width
        self.pallet_y = pallet_y  # Length
        self.max_height = max_height
        self.margin = margin
        
        # Track placed boxes
        self.placed_boxes = []  # (x, y, z, w, l, h, quaternion)  # Note: w, l instead of l, w
        self.box_count = 0
        
        # Visualization attributes
        self.fig = None
        self.ax = None
        self.show_margins = True
    
    def reset(self):
        """Reset the optimizer state"""
        self.placed_boxes = []
        self.box_count = 0
        
        # Clear visualization if it exists
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def add_box(self, width, length, height, change_stack_allowed=False):
        """
        Add a new box to the stack
        
        Args:
            width: Width of the box (x dimension)
            length: Length of the box (y dimension)
            height: Height of the box (z dimension)
            change_stack_allowed: Whether to allow rearranging existing boxes
            
        Returns:
            success: Whether placement succeeded
            position: (x, y, z) position for the box
            quaternion: Orientation of the box
        """
        if change_stack_allowed:
            # We can rearrange all boxes, including previously placed ones
            boxes = self.placed_boxes.copy()
            boxes.append((width, length, height))
            
            # Solve from scratch with all boxes
            dimensions = [(b[0], b[1], b[2]) for b in boxes]
            success, positions, rotations = self._optimize_stack(dimensions)
            
            if success:
                # Update all box positions
                self.placed_boxes = []
                for i in range(len(boxes)):
                    box_dims = dimensions[i]
                    x, y, z = positions[i]
                    rotation = rotations[i]
                    
                    # Create quaternion from rotation (0 = no rotation, 1 = 90 deg rotation)
                    if rotation:
                        # 90 degree rotation around z-axis
                        quat = euler2quat(0, 0, np.pi/2)
                        box_width, box_length = box_dims[1], box_dims[0]  # Swap width/length
                    else:
                        quat = euler2quat(0, 0, 0)  # No rotation
                        box_width, box_length = box_dims[0], box_dims[1]
                    
                    self.placed_boxes.append((x, y, z, box_width, box_length, box_dims[2], quat))
                
                # Return position and quaternion for the newest box
                newest_pos = positions[-1]
                newest_rot = rotations[-1]
                
                if newest_rot:
                    quat = euler2quat(0, 0, np.pi/2)
                else:
                    quat = euler2quat(0, 0, 0)
                
                self.box_count += 1
                return True, newest_pos, quat
            else:
                return False, None, None
        else:
            # Cannot rearrange existing boxes, only place the new one
            boxes = [b for b in self.placed_boxes]  # Fixed positions
            
            # Solve for the new box only
            success, position, rotation = self._optimize_new_box_placement(width, length, height)
            
            if success:
                # Create quaternion from rotation
                if rotation:
                    quat = euler2quat(0, 0, np.pi/2)
                    box_width, box_length = length, width  # Swap width/length
                else:
                    quat = euler2quat(0, 0, 0)  # No rotation
                    box_width, box_length = width, length
                
                self.placed_boxes.append((position[0], position[1], position[2], 
                                         box_width, box_length, height, quat))
                self.box_count += 1
                return True, position, quat
            else:
                return False, None, None
    
    def _optimize_stack(self, boxes):
        """
        Optimize the stacking of all boxes from scratch
        
        Args:
            boxes: List of (width, length, height) tuples
            
        Returns:
            success: Whether optimization succeeded
            positions: List of (x, y, z) positions
            rotations: List of rotation indicators (0 = no rotation, 1 = 90 deg rotation)
        """
        n = len(boxes)
        model = cp_model.CpModel()
        
        x, y, z = [], [], []
        r = []  # rotation variable
        width_vars, length_vars = [], []
        
        for i in range(n):
            w, l, h = boxes[i]  # width, length, height
            # Add margin to x and y lower bounds
            x_i = model.NewIntVar(self.margin, self.pallet_x - self.margin, f'x_{i}')
            y_i = model.NewIntVar(self.margin, self.pallet_y - self.margin, f'y_{i}')
            z_i = model.NewIntVar(0, self.max_height, f'z_{i}')
            r_i = model.NewBoolVar(f'r_{i}')  # 0 = no rotation, 1 = 90 deg rotation
            
            w_i = model.NewIntVar(0, self.pallet_x, f'w_{i}')
            l_i = model.NewIntVar(0, self.pallet_y, f'l_{i}')
            model.Add(w_i == w).OnlyEnforceIf(r_i.Not())
            model.Add(w_i == l).OnlyEnforceIf(r_i)
            model.Add(l_i == l).OnlyEnforceIf(r_i.Not())
            model.Add(l_i == w).OnlyEnforceIf(r_i)
            
            # Make sure box fits on pallet with margin
            model.Add(x_i + w_i <= self.pallet_x - self.margin)
            model.Add(y_i + l_i <= self.pallet_y - self.margin)
            
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
            r.append(r_i)
            width_vars.append(w_i)
            length_vars.append(l_i)
        
        # No-overlap constraints with margins
        for i in range(n):
            for j in range(i + 1, n):
                b1 = model.NewBoolVar(f'no_overlap_x1_{i}_{j}')
                b2 = model.NewBoolVar(f'no_overlap_x2_{i}_{j}')
                b3 = model.NewBoolVar(f'no_overlap_y1_{i}_{j}')
                b4 = model.NewBoolVar(f'no_overlap_y2_{i}_{j}')
                b5 = model.NewBoolVar(f'no_overlap_z1_{i}_{j}')
                b6 = model.NewBoolVar(f'no_overlap_z2_{i}_{j}')
                
                # Add margin to no-overlap constraints
                model.Add(x[i] + width_vars[i] + self.margin <= x[j]).OnlyEnforceIf(b1)
                model.Add(x[i] + width_vars[i] + self.margin > x[j]).OnlyEnforceIf(b1.Not())
                
                model.Add(x[j] + width_vars[j] + self.margin <= x[i]).OnlyEnforceIf(b2)
                model.Add(x[j] + width_vars[j] + self.margin > x[i]).OnlyEnforceIf(b2.Not())
                
                model.Add(y[i] + length_vars[i] + self.margin <= y[j]).OnlyEnforceIf(b3)
                model.Add(y[i] + length_vars[i] + self.margin > y[j]).OnlyEnforceIf(b3.Not())
                
                model.Add(y[j] + length_vars[j] + self.margin <= y[i]).OnlyEnforceIf(b4)
                model.Add(y[j] + length_vars[j] + self.margin > y[i]).OnlyEnforceIf(b4.Not())
                
                # Don't add margins to Z axis overlap to maintain physical stacking
                model.Add(z[i] + boxes[i][2] <= z[j]).OnlyEnforceIf(b5)
                model.Add(z[i] + boxes[i][2] > z[j]).OnlyEnforceIf(b5.Not())
                
                model.Add(z[j] + boxes[j][2] <= z[i]).OnlyEnforceIf(b6)
                model.Add(z[j] + boxes[j][2] > z[i]).OnlyEnforceIf(b6.Not())
                
                model.AddBoolOr([b1, b2, b3, b4, b5, b6])
        
        # Enhanced stability constraints
        for i in range(n):
            support_conditions = []
            
            # Support on pallet - only for boxes at z=0
            support_on_pallet = model.NewBoolVar(f'support_pallet_{i}')
            model.Add(z[i] == 0).OnlyEnforceIf(support_on_pallet)
            model.Add(z[i] != 0).OnlyEnforceIf(support_on_pallet.Not())
            support_conditions.append(support_on_pallet)
            
            # Find potential supporting boxes (must be directly below)
            potential_supporters = []
            for j in range(n):
                if i == j:
                    continue
                    
                # Create a variable to determine if box j is at the correct z-level to support box i
                correct_z_level = model.NewBoolVar(f'correct_z_level_{i}_{j}')
                model.Add(z[i] == z[j] + boxes[j][2]).OnlyEnforceIf(correct_z_level)
                model.Add(z[i] != z[j] + boxes[j][2]).OnlyEnforceIf(correct_z_level.Not())
                
                potential_supporters.append((j, correct_z_level))
            
            # For each box at the correct z-level, check if it provides sufficient support
            for j, correct_z_level in potential_supporters:
                support_area = model.NewBoolVar(f'support_area_{i}_{j}')
                
                # Calculate overlap area variables
                x_overlap_start = model.NewIntVar(0, self.pallet_x, f'x_overlap_start_{i}_{j}')
                x_overlap_end = model.NewIntVar(0, self.pallet_x, f'x_overlap_end_{i}_{j}')
                y_overlap_start = model.NewIntVar(0, self.pallet_y, f'y_overlap_start_{i}_{j}')
                y_overlap_end = model.NewIntVar(0, self.pallet_y, f'y_overlap_end_{i}_{j}')
                
                # Compute overlap in x-direction
                model.AddMaxEquality(x_overlap_start, [x[i], x[j]])
                model.AddMinEquality(x_overlap_end, [x[i] + width_vars[i], x[j] + width_vars[j]])
                
                # Compute overlap in y-direction
                model.AddMaxEquality(y_overlap_start, [y[i], y[j]])
                model.AddMinEquality(y_overlap_end, [y[i] + length_vars[i], y[j] + length_vars[j]])
                
                # Calculate overlap area
                x_overlap = model.NewIntVar(-self.pallet_x, self.pallet_x, f'x_overlap_{i}_{j}')
                y_overlap = model.NewIntVar(-self.pallet_y, self.pallet_y, f'y_overlap_{i}_{j}')
                model.Add(x_overlap == x_overlap_end - x_overlap_start)
                model.Add(y_overlap == y_overlap_end - y_overlap_start)
                
                # Require positive overlap
                positive_x_overlap = model.NewBoolVar(f'positive_x_overlap_{i}_{j}')
                positive_y_overlap = model.NewBoolVar(f'positive_y_overlap_{i}_{j}')
                model.Add(x_overlap > 0).OnlyEnforceIf(positive_x_overlap)
                model.Add(x_overlap <= 0).OnlyEnforceIf(positive_x_overlap.Not())
                model.Add(y_overlap > 0).OnlyEnforceIf(positive_y_overlap)
                model.Add(y_overlap <= 0).OnlyEnforceIf(positive_y_overlap.Not())
                
                # Require both overlaps to be positive for support
                model.AddBoolAnd([positive_x_overlap, positive_y_overlap]).OnlyEnforceIf(support_area)
                model.AddBoolOr([positive_x_overlap.Not(), positive_y_overlap.Not()]).OnlyEnforceIf(support_area.Not())
                
                # Require sufficient support area - at least 50% of box area must be supported
                min_overlap_x = model.NewIntVar(0, self.pallet_x, f'min_overlap_x_{i}_{j}')
                # Using multiplication instead of division for IntVar
                model.Add(2 * min_overlap_x == width_vars[i])
                
                min_overlap_y = model.NewIntVar(0, self.pallet_y, f'min_overlap_y_{i}_{j}')
                # Using multiplication instead of division for IntVar
                model.Add(2 * min_overlap_y == length_vars[i])
                
                sufficient_x_overlap = model.NewBoolVar(f'sufficient_x_overlap_{i}_{j}')
                sufficient_y_overlap = model.NewBoolVar(f'sufficient_y_overlap_{i}_{j}')
                model.Add(x_overlap >= min_overlap_x).OnlyEnforceIf(sufficient_x_overlap)
                model.Add(x_overlap < min_overlap_x).OnlyEnforceIf(sufficient_x_overlap.Not())
                model.Add(y_overlap >= min_overlap_y).OnlyEnforceIf(sufficient_y_overlap)
                model.Add(y_overlap < min_overlap_y).OnlyEnforceIf(sufficient_y_overlap.Not())
                
                # Define overall sufficient support
                sufficient_support = model.NewBoolVar(f'sufficient_support_{i}_{j}')
                model.AddBoolAnd([support_area, sufficient_x_overlap, sufficient_y_overlap]).OnlyEnforceIf(sufficient_support)
                model.AddBoolOr([support_area.Not(), sufficient_x_overlap.Not(), sufficient_y_overlap.Not()]).OnlyEnforceIf(sufficient_support.Not())
                
                # A box supports another only if it's at the correct z-level and provides sufficient support
                supports = model.NewBoolVar(f'supports_{i}_{j}')
                model.AddBoolAnd([correct_z_level, sufficient_support]).OnlyEnforceIf(supports)
                model.AddBoolOr([correct_z_level.Not(), sufficient_support.Not()]).OnlyEnforceIf(supports.Not())
                
                support_conditions.append(supports)
            
            # Each box must be supported by either the pallet or at least one other box
            model.AddBoolOr(support_conditions)
        
        # Objective: minimize total height and maximize stability
        max_z = model.NewIntVar(0, self.max_height, 'max_z')
        
        # Track the total unsupported area across all boxes
        total_unsupported_area = model.NewIntVar(0, self.pallet_x * self.pallet_y * n, 'total_unsupported_area')
        unsupported_areas = []
        
        for i in range(n):
            model.Add(max_z >= z[i] + boxes[i][2])
            
            # Calculate total base area of the box
            base_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'base_area_{i}')
            model.Add(base_area == width_vars[i] * length_vars[i])
            
            # For boxes on the pallet (z=0), the support area is the full base area
            on_pallet = model.NewBoolVar(f'on_pallet_{i}')
            model.Add(z[i] == 0).OnlyEnforceIf(on_pallet)
            model.Add(z[i] != 0).OnlyEnforceIf(on_pallet.Not())
            
            # For boxes not on the pallet, calculate support areas from boxes below
            supported_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'supported_area_{i}')
            
            # If on pallet, supported area is same as base area
            model.Add(supported_area == base_area).OnlyEnforceIf(on_pallet)
            
            # If not on pallet, calculate supported area from boxes below
            if i > 0:  # Only needed if there are boxes to support this one
                support_areas = []
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Check if box j is directly below box i
                    is_below = model.NewBoolVar(f'is_below_{i}_{j}')
                    model.Add(z[i] == z[j] + boxes[j][2]).OnlyEnforceIf(is_below)
                    model.Add(z[i] != z[j] + boxes[j][2]).OnlyEnforceIf(is_below.Not())
                    
                    # Calculate overlap area
                    overlap_area = self._calculate_overlap_area(model, i, j, x, y, width_vars, length_vars)
                    
                    # This area only counts if box j is directly below box i
                    effective_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'effective_area_{i}_{j}')
                    model.Add(effective_area == overlap_area).OnlyEnforceIf(is_below)
                    model.Add(effective_area == 0).OnlyEnforceIf(is_below.Not())
                    
                    support_areas.append(effective_area)
                
                # Add all support areas (will only include boxes directly below)
                if support_areas:
                    total_support = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'total_support_{i}')
                    model.Add(total_support == sum(support_areas))
                    model.Add(supported_area == total_support).OnlyEnforceIf(on_pallet.Not())
            
            # Calculate unsupported area
            unsupported_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'unsupported_area_{i}')
            model.Add(unsupported_area == base_area - supported_area)
            
            # Ensure this doesn't go negative
            model.Add(unsupported_area >= 0)
            
            unsupported_areas.append(unsupported_area)
        
        # Sum all unsupported areas
        model.Add(total_unsupported_area == sum(unsupported_areas))
        
        # Calculate edge proximity scores
        edge_distance_sum = model.NewIntVar(0, self.pallet_x * self.pallet_y * n * 2, 'edge_distance_sum')
        edge_distances = []

        for i in range(n):
            # Calculate distance to each edge (higher is worse - further from edge)
            dist_to_left = model.NewIntVar(0, self.pallet_x, f'dist_left_{i}')
            dist_to_right = model.NewIntVar(0, self.pallet_x, f'dist_right_{i}')
            dist_to_bottom = model.NewIntVar(0, self.pallet_y, f'dist_bottom_{i}')
            dist_to_top = model.NewIntVar(0, self.pallet_y, f'dist_top_{i}')
            
            # Distance to left edge is just x coordinate
            model.Add(dist_to_left == x[i])
            
            # Distance to right edge is (pallet_x - (x + width))
            model.Add(dist_to_right == self.pallet_x - (x[i] + width_vars[i]))
            
            # Distance to bottom edge is just y coordinate
            model.Add(dist_to_bottom == y[i])
            
            # Distance to top edge is (pallet_y - (y + length))
            model.Add(dist_to_top == self.pallet_y - (y[i] + length_vars[i]))
            
            # Calculate the sum of the TWO smallest distances to encourage corner placement
            # First find min distances for each axis
            min_x_dist = model.NewIntVar(0, self.pallet_x, f'min_x_dist_{i}')
            min_y_dist = model.NewIntVar(0, self.pallet_y, f'min_y_dist_{i}')
            
            model.AddMinEquality(min_x_dist, [dist_to_left, dist_to_right])
            model.AddMinEquality(min_y_dist, [dist_to_bottom, dist_to_top])
            
            # Sum the minimum distances in each axis - this encourages corner placement
            two_edge_distance = model.NewIntVar(0, self.pallet_x + self.pallet_y, f'two_edge_distance_{i}')
            model.Add(two_edge_distance == min_x_dist + min_y_dist)
            
            # Add to total edge distance
            edge_distances.append(two_edge_distance)

        # Sum all edge distances
        model.Add(edge_distance_sum == sum(edge_distances))

        # Weights for balancing objectives
        stability_weight = 10  # Weight for unsupported area
        edge_weight = 5        # Weight for distance from edges

        # Combined objective: minimize height, unsupported area, and distance from edges
        model.Minimize(max_z + stability_weight * total_unsupported_area + edge_weight * edge_distance_sum)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # Limit solving time
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            positions = []
            rotations = []
            for i in range(n):
                pos = (solver.Value(x[i]), solver.Value(y[i]), solver.Value(z[i]))
                rot = solver.Value(r[i])
                positions.append(pos)
                rotations.append(rot)
            return True, positions, rotations
        else:
            return False, None, None
    
    def _optimize_new_box_placement(self, width, length, height):
        """
        Optimize placement of a new box with the existing stack
        
        Args:
            width: Width of the box (x dimension)
            length: Length of the box (y dimension)
            height: Height of the box
            
        Returns:
            success: Whether placement succeeded
            position: (x, y, z) position for the box
            rotation: 0 = no rotation, 1 = 90 deg rotation
        """
        model = cp_model.CpModel()

        # First try to place the box on the ground level (z=0)
        # If that fails, allow stacking on other boxes
        
        # Try ground placement first
        ground_model = cp_model.CpModel()
        
        # Variables for the new box - only on ground level
        x_new = ground_model.NewIntVar(self.margin, self.pallet_x - self.margin, 'x_new')
        y_new = ground_model.NewIntVar(self.margin, self.pallet_y - self.margin, 'y_new')
        # Force z=0 for ground placement
        z_new = ground_model.NewConstant(0)
        r_new = ground_model.NewBoolVar('r_new')  # 0 = no rotation, 1 = 90 deg rotation
        
        # Define width and length variables based on rotation
        w_new = ground_model.NewIntVar(0, self.pallet_x, 'w_new')
        l_new = ground_model.NewIntVar(0, self.pallet_y, 'l_new')
        ground_model.Add(w_new == width).OnlyEnforceIf(r_new.Not())
        ground_model.Add(w_new == length).OnlyEnforceIf(r_new)
        ground_model.Add(l_new == length).OnlyEnforceIf(r_new.Not())
        ground_model.Add(l_new == width).OnlyEnforceIf(r_new)
        
        # Make sure box fits on pallet with margin
        ground_model.Add(x_new + w_new <= self.pallet_x - self.margin)
        ground_model.Add(y_new + l_new <= self.pallet_y - self.margin)
        
        # No-overlap constraints with existing boxes on ground level
        for i, box in enumerate(self.placed_boxes):
            x_i, y_i, z_i, w_i, l_i, h_i, _ = box
            
            # Only check ground-level boxes
            if z_i == 0:
                b1 = ground_model.NewBoolVar(f'no_overlap_x1_new_{i}')
                b2 = ground_model.NewBoolVar(f'no_overlap_x2_new_{i}')
                b3 = ground_model.NewBoolVar(f'no_overlap_y1_new_{i}')
                b4 = ground_model.NewBoolVar(f'no_overlap_y2_new_{i}')
                
                # Add margin to no-overlap constraints
                ground_model.Add(x_new + w_new + self.margin <= x_i).OnlyEnforceIf(b1)
                ground_model.Add(x_new + w_new + self.margin > x_i).OnlyEnforceIf(b1.Not())
                
                ground_model.Add(x_i + w_i + self.margin <= x_new).OnlyEnforceIf(b2)
                ground_model.Add(x_i + w_i + self.margin > x_new).OnlyEnforceIf(b2.Not())
                
                ground_model.Add(y_new + l_new + self.margin <= y_i).OnlyEnforceIf(b3)
                ground_model.Add(y_new + l_new + self.margin > y_i).OnlyEnforceIf(b3.Not())
                
                ground_model.Add(y_i + l_i + self.margin <= y_new).OnlyEnforceIf(b4)
                ground_model.Add(y_i + l_i + self.margin > y_new).OnlyEnforceIf(b4.Not())
                
                ground_model.AddBoolOr([b1, b2, b3, b4])
        
        # Simple objective for ground placement: minimize distance from origin
        ground_model.Minimize(x_new + y_new)
        
        ground_solver = cp_model.CpSolver()
        ground_solver.parameters.max_time_in_seconds = 2.0  # Shorter timeout
        ground_status = ground_solver.Solve(ground_model)
        
        # Add better logging and validation for the ground placement solution
        if ground_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Found potential ground-level placement solution")
            
            # Get the solution values
            solution_x = ground_solver.Value(x_new)
            solution_y = ground_solver.Value(y_new)
            solution_rot = ground_solver.Value(r_new)
            
            # Determine actual width and length based on rotation
            solution_width = length if solution_rot else width
            solution_length = width if solution_rot else length
            
            # Validate solution against existing boxes using a more direct approach
            valid_solution = True
            for i, box in enumerate(self.placed_boxes):
                x_i, y_i, z_i, w_i, l_i, h_i, _ = box
                if z_i == 0:  # Only check ground-level boxes
                    # Print detailed box positions for debugging
                    print(f"Validating against box {i}:")
                    print(f"  New box: corner=({solution_x}, {solution_y}), dims={solution_width}x{solution_length}")
                    print(f"  Existing box: corner=({x_i}, {y_i}), dims={w_i}x{l_i}")
                    
                    # Calculate the ranges for both boxes
                    new_box_x_min, new_box_x_max = solution_x, solution_x + solution_width
                    new_box_y_min, new_box_y_max = solution_y, solution_y + solution_length
                    
                    existing_box_x_min, existing_box_x_max = x_i, x_i + w_i
                    existing_box_y_min, existing_box_y_max = y_i, y_i + l_i
                    
                    # Check if the boxes are far enough apart
                    x_separation = (new_box_x_min >= existing_box_x_max + self.margin) or (existing_box_x_min >= new_box_x_max + self.margin)
                    y_separation = (new_box_y_min >= existing_box_y_max + self.margin) or (existing_box_y_min >= new_box_y_max + self.margin)
                    
                    # If boxes are not separated in both X and Y, there's an overlap
                    if not (x_separation or y_separation):
                        print(f"Overlap detected between new box and box {i}!")
                        print(f"  X overlap: {not x_separation} (new={new_box_x_min}-{new_box_x_max}, existing={existing_box_x_min}-{existing_box_x_max})")
                        print(f"  Y overlap: {not y_separation} (new={new_box_y_min}-{new_box_y_max}, existing={existing_box_y_min}-{existing_box_y_max})")
                        valid_solution = False
                        break
                    else:
                        print(f"  No overlap - boxes are separated in {'X' if x_separation else 'Y'} dimension")
            
            if valid_solution:
                print("Solution validated - no overlaps with existing boxes")
                return True, (solution_x, solution_y, 0), solution_rot
            else:
                print("Invalid ground placement - overlaps detected")
                # Continue to try non-ground placement
        else:
            print(f"Ground placement failed with status: {ground_status}")
        
        print("Trying non-ground placement...")
        
        # Rest of the method remains the same (non-ground placement logic)
        model = cp_model.CpModel()
        
        # Variables for the new box
        # Add margin to x and y boundaries
        x_new = model.NewIntVar(self.margin, self.pallet_x - self.margin, 'x_new')
        y_new = model.NewIntVar(self.margin, self.pallet_y - self.margin, 'y_new')
        z_new = model.NewIntVar(0, self.max_height, 'z_new')
        r_new = model.NewBoolVar('r_new')  # 0 = no rotation, 1 = 90 deg rotation

        
        w_new = model.NewIntVar(0, self.pallet_x, 'w_new')
        l_new = model.NewIntVar(0, self.pallet_y, 'l_new')
        model.Add(l_new == length).OnlyEnforceIf(r_new.Not())
        model.Add(w_new == width).OnlyEnforceIf(r_new)
        model.Add(w_new == width).OnlyEnforceIf(r_new.Not())
        model.Add(l_new == length).OnlyEnforceIf(r_new)
        
        # Make sure box fits on pallet with margin
        model.Add(x_new + w_new <= self.pallet_x - self.margin)
        model.Add(y_new + l_new <= self.pallet_y - self.margin)
        
        # No-overlap constraints with existing boxes, with margins
        for i, box in enumerate(self.placed_boxes):
            x_i, y_i, z_i, l_i, w_i, h_i, _ = box
            
            b1 = model.NewBoolVar(f'no_overlap_x1_new_{i}')
            b2 = model.NewBoolVar(f'no_overlap_x2_new_{i}')
            b3 = model.NewBoolVar(f'no_overlap_y1_new_{i}')
            b4 = model.NewBoolVar(f'no_overlap_y2_new_{i}')
            b5 = model.NewBoolVar(f'no_overlap_z1_new_{i}')
            b6 = model.NewBoolVar(f'no_overlap_z2_new_{i}')
            
            # Add margin to no-overlap constraints
            model.Add(x_new + w_new + self.margin <= x_i).OnlyEnforceIf(b1)
            model.Add(x_new + w_new + self.margin > x_i).OnlyEnforceIf(b1.Not())
            
            model.Add(x_i + l_i + self.margin <= x_new).OnlyEnforceIf(b2)
            model.Add(x_i + l_i + self.margin > x_new).OnlyEnforceIf(b2.Not())
            
            model.Add(y_new + l_new + self.margin <= y_i).OnlyEnforceIf(b3)
            model.Add(y_new + l_new + self.margin > y_i).OnlyEnforceIf(b3.Not())
            
            model.Add(y_i + w_i + self.margin <= y_new).OnlyEnforceIf(b4)
            model.Add(y_i + w_i + self.margin > y_new).OnlyEnforceIf(b4.Not())
            
            # No margin for Z axis to maintain physical stacking
            model.Add(z_new + height <= z_i).OnlyEnforceIf(b5)
            model.Add(z_new + height > z_i).OnlyEnforceIf(b5.Not())
            
            model.Add(z_i + h_i <= z_new).OnlyEnforceIf(b6)
            model.Add(z_i + h_i > z_new).OnlyEnforceIf(b6.Not())
            
            model.AddBoolOr([b1, b2, b3, b4, b5, b6])
        
        # Enhanced stability constraints for single box placement
        support_conditions = []
        
        # Support on pallet - only for boxes directly on the pallet
        support_on_pallet = model.NewBoolVar('support_pallet_new')
        model.Add(z_new == 0).OnlyEnforceIf(support_on_pallet)
        model.Add(z_new != 0).OnlyEnforceIf(support_on_pallet.Not())
        support_conditions.append(support_on_pallet)
        
        # Find potential supporting boxes (must be directly below)
        potential_supporters = []
        for i, box in enumerate(self.placed_boxes):
            x_i, y_i, z_i, l_i, w_i, h_i, _ = box
            
            # Determine if this box is at the correct height to support the new box
            correct_z_level = model.NewBoolVar(f'correct_z_level_new_{i}')
            model.Add(z_new == z_i + h_i).OnlyEnforceIf(correct_z_level)
            model.Add(z_new != z_i + h_i).OnlyEnforceIf(correct_z_level.Not())
            
            potential_supporters.append((i, box, correct_z_level))
        
        # Calculate base area of the new box
        base_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, 'base_area_new')
        model.AddMultiplicationEquality(base_area, [w_new, l_new])
        
        # For the new box, calculate support areas from boxes below
        supported_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, 'supported_area_new')
        
        # If on pallet, supported area is same as base area
        model.Add(supported_area == base_area).OnlyEnforceIf(support_on_pallet)
        
        # If not on pallet, calculate supported area from boxes below
        support_areas = []
        
        # For each potential supporter, determine if there's sufficient support
        for i, box, correct_z_level in potential_supporters:
            x_i, y_i, z_i, l_i, w_i, h_i, _ = box
            
            # Calculate overlap area
            x_overlap_start = model.NewIntVar(0, self.pallet_x, f'x_overlap_start_new_{i}')
            x_overlap_end = model.NewIntVar(0, self.pallet_x, f'x_overlap_end_new_{i}')
            y_overlap_start = model.NewIntVar(0, self.pallet_y, f'y_overlap_start_new_{i}')
            y_overlap_end = model.NewIntVar(0, self.pallet_y, f'y_overlap_end_new_{i}')
            
            # Compute overlap in x-direction
            model.AddMaxEquality(x_overlap_start, [x_new, x_i])
            model.AddMinEquality(x_overlap_end, [x_new + w_new, x_i + l_i])
            
            # Compute overlap in y-direction
            model.AddMaxEquality(y_overlap_start, [y_new, y_i])
            model.AddMinEquality(y_overlap_end, [y_new + l_new, y_i + w_i])
            
            # Calculate overlap dimensions
            x_overlap = model.NewIntVar(-self.pallet_x, self.pallet_x, f'x_overlap_new_{i}')
            y_overlap = model.NewIntVar(-self.pallet_y, self.pallet_y, f'y_overlap_new_{i}')
            model.Add(x_overlap == x_overlap_end - x_overlap_start)
            model.Add(y_overlap == y_overlap_end - y_overlap_start)
            
            # Handle negative overlaps (no overlap case)
            x_overlap_pos = model.NewIntVar(0, self.pallet_x, f'x_overlap_pos_new_{i}')
            y_overlap_pos = model.NewIntVar(0, self.pallet_y, f'y_overlap_pos_new_{i}')
            
            # x_overlap_pos = max(0, x_overlap)
            model.AddMaxEquality(x_overlap_pos, [x_overlap, model.NewConstant(0)])
            # y_overlap_pos = max(0, y_overlap)
            model.AddMaxEquality(y_overlap_pos, [y_overlap, model.NewConstant(0)])
            
            # Calculate area = x_overlap_pos * y_overlap_pos
            overlap_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'overlap_area_new_{i}')
            model.AddMultiplicationEquality(overlap_area, [x_overlap_pos, y_overlap_pos])
            
            # This area only counts if box i is directly below the new box
            effective_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'effective_area_new_{i}')
            model.Add(effective_area == overlap_area).OnlyEnforceIf(correct_z_level)
            model.Add(effective_area == 0).OnlyEnforceIf(correct_z_level.Not())
            
            support_areas.append(effective_area)
            
            # Check for positive overlap
            positive_x_overlap = model.NewBoolVar(f'positive_x_overlap_new_{i}')
            positive_y_overlap = model.NewBoolVar(f'positive_y_overlap_new_{i}')
            model.Add(x_overlap > 0).OnlyEnforceIf(positive_x_overlap)
            model.Add(x_overlap <= 0).OnlyEnforceIf(positive_x_overlap.Not())
            model.Add(y_overlap > 0).OnlyEnforceIf(positive_y_overlap)
            model.Add(y_overlap <= 0).OnlyEnforceIf(positive_y_overlap.Not())
            
            # Both dimensions must have positive overlap
            any_overlap = model.NewBoolVar(f'any_overlap_new_{i}')
            model.AddBoolAnd([positive_x_overlap, positive_y_overlap]).OnlyEnforceIf(any_overlap)
            model.AddBoolOr([positive_x_overlap.Not(), positive_y_overlap.Not()]).OnlyEnforceIf(any_overlap.Not())
            
            # Require sufficient support area - at least 50% of the new box's footprint
            min_overlap_x = model.NewIntVar(0, self.pallet_x, f'min_overlap_x_new_{i}')
            # Using multiplication instead of division for IntVar
            model.Add(2 * min_overlap_x == w_new)
            
            min_overlap_y = model.NewIntVar(0, self.pallet_y, f'min_overlap_y_new_{i}')
            # Using multiplication instead of division for IntVar
            model.Add(2 * min_overlap_y == l_new)
            
            sufficient_x_overlap = model.NewBoolVar(f'sufficient_x_overlap_new_{i}')
            sufficient_y_overlap = model.NewBoolVar(f'sufficient_y_overlap_new_{i}')
            model.Add(x_overlap >= min_overlap_x).OnlyEnforceIf(sufficient_x_overlap)
            model.Add(x_overlap < min_overlap_x).OnlyEnforceIf(sufficient_x_overlap.Not())
            model.Add(y_overlap >= min_overlap_y).OnlyEnforceIf(sufficient_y_overlap)
            model.Add(y_overlap < min_overlap_y).OnlyEnforceIf(sufficient_y_overlap.Not())
            
            # Both dimensions must have sufficient overlap
            sufficient_support = model.NewBoolVar(f'sufficient_support_new_{i}')
            model.AddBoolAnd([sufficient_x_overlap, sufficient_y_overlap]).OnlyEnforceIf(sufficient_support)
            model.AddBoolOr([sufficient_x_overlap.Not(), sufficient_y_overlap.Not()]).OnlyEnforceIf(sufficient_support.Not())
            
            # Box i supports the new box if:
            # 1. It's at the correct z-level (directly below)
            # 2. It provides sufficient support area
            supports = model.NewBoolVar(f'supports_new_{i}')
            model.AddBoolAnd([correct_z_level, sufficient_support]).OnlyEnforceIf(supports)
            model.AddBoolOr([correct_z_level.Not(), sufficient_support.Not()]).OnlyEnforceIf(supports.Not())
            
            support_conditions.append(supports)
        
        # The new box must be supported by either the pallet or one or more boxes directly beneath it
        model.AddBoolOr(support_conditions)
        
        # Sum all support areas for boxes directly below
        if support_areas:
            total_support = model.NewIntVar(0, self.pallet_x * self.pallet_y, 'total_support_new')
            model.Add(total_support == sum(support_areas))
            model.Add(supported_area == total_support).OnlyEnforceIf(support_on_pallet.Not())
        
        # Calculate unsupported area
        unsupported_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, 'unsupported_area_new')
        model.Add(unsupported_area == base_area - supported_area)
        
        # Ensure this doesn't go negative
        model.Add(unsupported_area >= 0)
        
        dist_to_left = model.NewIntVar(0, self.pallet_x, 'dist_left_new')
        dist_to_right = model.NewIntVar(0, self.pallet_x, 'dist_right_new')
        dist_to_bottom = model.NewIntVar(0, self.pallet_y, 'dist_bottom_new')
        dist_to_top = model.NewIntVar(0, self.pallet_y, 'dist_top_new')

        # Distance to left edge is just x coordinate
        model.Add(dist_to_left == x_new)

        # Distance to right edge is (pallet_x - (x + length))
        model.Add(dist_to_right == self.pallet_x - (x_new + w_new))

        # Distance to bottom edge is just y coordinate
        model.Add(dist_to_bottom == y_new)

        # Distance to top edge is (pallet_y - (y + width))
        model.Add(dist_to_top == self.pallet_y - (y_new + l_new))

        # Calculate the minimum distance in each axis
        min_x_dist = model.NewIntVar(0, self.pallet_x, 'min_x_dist_new')
        min_y_dist = model.NewIntVar(0, self.pallet_y, 'min_y_dist_new')

        model.AddMinEquality(min_x_dist, [dist_to_left, dist_to_right])
        model.AddMinEquality(min_y_dist, [dist_to_bottom, dist_to_top])

        # Sum the minimum distances in each axis to encourage corner placement
        edge_distance = model.NewIntVar(0, self.pallet_x + self.pallet_y, 'two_edge_distance_new')
        model.Add(edge_distance == min_x_dist + min_y_dist)

        # Weights for balancing objectives
        height_weight = 15     # Increased weight for height minimization
        stability_weight = 10  # Weight for unsupported area
        edge_weight = 5        # Weight for distance from edges

        # Combined objective: minimize height, unsupported area, and edge distance
        model.Minimize(height_weight * z_new + stability_weight * unsupported_area + edge_weight * edge_distance)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # Limit solving time
        status = solver.Solve(model)

        # Add detailed logging
        status_name = "UNKNOWN"
        if status == cp_model.OPTIMAL:
            status_name = "OPTIMAL"
        elif status == cp_model.FEASIBLE:
            status_name = "FEASIBLE"
        elif status == cp_model.INFEASIBLE:
            status_name = "INFEASIBLE"
        elif status == cp_model.MODEL_INVALID:
            status_name = "MODEL_INVALID"
        elif status == cp_model.UNKNOWN:
            status_name = "UNKNOWN (could be timeout)"
        
        print(f"Solver status: {status_name}")
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return True, (solver.Value(x_new), solver.Value(y_new), solver.Value(z_new)), solver.Value(r_new)
        else:
            # Print more details about the constraints
            print(f"Failed to place box with dimensions {width}x{length}x{height}")
            print(f"Current stack has {len(self.placed_boxes)} boxes")
            for i, box in enumerate(self.placed_boxes):
                x, y, z, w, l, h, _ = box
                print(f"Box {i}: at ({x}, {y}, {z}) with dimensions {w}x{l}x{h}")
            return False, None, None
    def _calculate_overlap_area(self, model, i, j, x, y, width_vars, length_vars):
        """
        Calculate the overlap area between box i and box j
        
        Args:
            model: CP model
            i, j: Indices of boxes
            x, y: Arrays of x and y position variables
            width_vars, length_vars: Arrays of box width and length variables
            
        Returns:
            overlap_area: Variable representing the overlap area
        """
        x_overlap_start = model.NewIntVar(0, self.pallet_x, f'x_overlap_start_{i}_{j}')
        x_overlap_end = model.NewIntVar(0, self.pallet_x, f'x_overlap_end_{i}_{j}')
        y_overlap_start = model.NewIntVar(0, self.pallet_y, f'y_overlap_start_{i}_{j}')
        y_overlap_end = model.NewIntVar(0, self.pallet_y, f'y_overlap_end_{i}_{j}')
        
        # Compute max of start positions (overlap start)
        model.AddMaxEquality(x_overlap_start, [x[i], x[j]])
        model.AddMaxEquality(y_overlap_start, [y[i], y[j]])
        
        # Compute min of end positions (overlap end)
        model.AddMinEquality(x_overlap_end, [x[i] + width_vars[i], x[j] + width_vars[j]])
        model.AddMinEquality(y_overlap_end, [y[i] + length_vars[i], y[j] + length_vars[j]])
        
        # Calculate overlap dimensions
        x_overlap = model.NewIntVar(-self.pallet_x, self.pallet_x, f'x_overlap_{i}_{j}')
        y_overlap = model.NewIntVar(-self.pallet_y, self.pallet_y, f'y_overlap_{i}_{j}')
        model.Add(x_overlap == x_overlap_end - x_overlap_start)
        model.Add(y_overlap == y_overlap_end - y_overlap_start)
        
        # Handle negative overlaps (no overlap case)
        x_overlap_pos = model.NewIntVar(0, self.pallet_x, f'x_overlap_pos_{i}_{j}')
        y_overlap_pos = model.NewIntVar(0, self.pallet_y, f'y_overlap_pos_{i}_{j}')
        
        # x_overlap_pos = max(0, x_overlap)
        model.AddMaxEquality(x_overlap_pos, [x_overlap, model.NewConstant(0)])
        # y_overlap_pos = max(0, y_overlap)
        model.AddMaxEquality(y_overlap_pos, [y_overlap, model.NewConstant(0)])
        
        # Calculate area = x_overlap_pos * y_overlap_pos
        overlap_area = model.NewIntVar(0, self.pallet_x * self.pallet_y, f'overlap_area_{i}_{j}')
        model.AddMultiplicationEquality(overlap_area, [x_overlap_pos, y_overlap_pos])
        
        return overlap_area

    # SIMPLIFIED VISUALIZATION - More direct approach
    def visualize_stack(self):
        """Visualize the current stack of boxes"""
        # Close any existing figure to avoid memory leaks
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create a new figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Configure the axes
        self.ax.set_xlim(0, max(self.pallet_x, self.pallet_y))
        self.ax.set_ylim(0, max(self.pallet_x, self.pallet_y))
        self.ax.set_zlim(0, self.max_height)
        self.ax.set_xlabel("X (Width)")
        self.ax.set_ylabel("Y (Length)")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Box Stack Visualization")
        
        # Draw pallet base
        self.ax.bar3d(0, 0, 0, self.pallet_x, self.pallet_y, 0.1, color='gray', alpha=0.3, shade=True)
        
        if not self.placed_boxes:
            plt.show()
            return
        
        # Get a colormap with enough colors for our boxes
        cmap = cm.get_cmap('tab20', max(len(self.placed_boxes), 1))
        
        # Plot each box
        for i, box in enumerate(self.placed_boxes):
            x, y, z, w, l, h, _ = box  # x, y, z, width, length, height, quaternion

            #Log dimensions and properties
            print(f'Box {i}: Position ({x}, {y}, {z}), Size ({w}, {l}, {h})')
            
            # Plot the box
            self._plot_box(self.ax, (x, y, z), (w, l, h), cmap(i))
            
            # Plot margins if enabled
            if self.show_margins:
                margin_origin = (x - self.margin, y - self.margin, z)
                margin_size = (w + 2*self.margin, l + 2*self.margin, h)
                self._plot_margin(self.ax, margin_origin, margin_size, cmap(i))
        
        # Add view controls
        self.ax.view_init(elev=30, azim=45)  # Default view
        
        # Show the figure with blocking=True to ensure it stays visible
        plt.show(block=False)
    
    def _plot_box(self, ax, origin, size, color):
        """
        Helper function to plot a box
        
        Args:
            ax: Matplotlib axis
            origin: (x, y, z) origin point
            size: (width, length, height) box dimensions
            color: Box color
        """
        x0, y0, z0 = origin
        dx, dy, dz = size  # width, length, height
        
        # Define the vertices
        verts = [
            [x0, y0, z0],
            [x0 + dx, y0, z0],
            [x0 + dx, y0 + dy, z0],
            [x0, y0 + dy, z0],
            [x0, y0, z0 + dz],
            [x0 + dx, y0, z0 + dz],
            [x0 + dx, y0 + dy, z0 + dz],
            [x0, y0 + dy, z0 + dz],
        ]
        
        # Define faces
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],  # bottom
            [verts[4], verts[5], verts[6], verts[7]],  # top
            [verts[0], verts[1], verts[5], verts[4]],  # front
            [verts[1], verts[2], verts[6], verts[5]],  # right
            [verts[2], verts[3], verts[7], verts[6]],  # back
            [verts[3], verts[0], verts[4], verts[7]],  # left
        ]
        print(f'Box vertices: {verts}')
        print(f'Box faces: {faces}')
        
        # Create poly collection
        collection = Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=0.8)
        ax.add_collection3d(collection)
    
    def _plot_margin(self, ax, origin, size, color):
        """
        Helper function to plot the safety margin boundaries
        
        Args:
            ax: Matplotlib axis
            origin: (x, y, z) origin point
            size: (width, length, height) margin box dimensions
            color: Box color
        """
        x0, y0, z0 = origin
        dx, dy, dz = size
        
        # Bottom frame
        ax.plot([x0, x0+dx], [y0, y0], [z0, z0], 'k--', alpha=0.3)
        ax.plot([x0, x0], [y0, y0+dy], [z0, z0], 'k--', alpha=0.3)
        ax.plot([x0+dx, x0+dx], [y0, y0+dy], [z0, z0], 'k--', alpha=0.3)
        ax.plot([x0, x0+dx], [y0+dy, y0+dy], [z0, z0], 'k--', alpha=0.3)
        
        # Top frame
        ax.plot([x0, x0+dx], [y0, y0], [z0+dz, z0+dz], 'k--', alpha=0.3)
        ax.plot([x0, x0], [y0, y0+dy], [z0+dz, z0+dz], 'k--', alpha=0.3)
        ax.plot([x0+dx, x0+dx], [y0, y0+dy], [z0+dz, z0+dz], 'k--', alpha=0.3)
        ax.plot([x0, x0+dx], [y0+dy, y0+dy], [z0+dz, z0+dz], 'k--', alpha=0.3)
        
        # Vertical edges
        ax.plot([x0, x0], [y0, y0], [z0, z0+dz], 'k--', alpha=0.3)
        ax.plot([x0+dx, x0+dx], [y0, y0], [z0, z0+dz], 'k--', alpha=0.3)
        ax.plot([x0+dx, x0+dx], [y0+dy, y0+dy], [z0, z0+dz], 'k--', alpha=0.3)
        ax.plot([x0, x0], [y0+dy, y0+dy], [z0, z0+dz], 'k--', alpha=0.3)


class BoxStackService(Node):
    def __init__(self):
        super().__init__('box_stack_optimizer_service')
        
        # Create the optimizer with the safety margin
        self.optimizer = BoxStackOptimizer(margin=10)  # Use 5 units as requested
        
        # Create the service
        self.srv = self.create_service(
            StackOptimizer, 
            'box_stack_optimizer', 
            self.handle_box_stack_request
        )

        self.box_state_publisher = self.create_publisher(BoxState, 'box_stack_state', 10)
        self.state_publish_timer = self.create_timer(0.1, self.publish_stack_state)

        
        # TF broadcaster for visualization
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Remove visualization timer - we'll call visualize directly when needed
        
        # Define unit conversion factors
        # If global coordinates are in meters and pallet dimensions were given in cm
        self.cm_to_m = 0.01  # Conversion factor from centimeters to meters
        
        # Define the optimizer unit to real-world scaling
        # The optimizer uses abstract units (4×2 pallet), but we need to scale to real dimensions
        # We'll calculate this based on the real-world pallet dimensions
        
        # Define the coordinate transformation from optimizer (local) to global coordinates
        # Pallet corners in global coordinates (in meters)
        top_left = np.array([0.5792, 0.4032, -0.7884])
        top_right = np.array([1.1630, 0.4828, -0.7889])
        bottom_left = np.array([0.6782, -0.3763, -0.7843])
        
        # Calculate vectors for the pallet coordinate system
        # X-axis: From top-left to bottom-left (as requested)
        x_axis = bottom_left - top_left
        # Y-axis: From top-left to top-right (as requested)
        y_axis = top_right - top_left
        
        # Create an orthogonal coordinate system
        x_axis_normalized = x_axis / np.linalg.norm(x_axis)
        # Make y_axis orthogonal to x_axis using Gram-Schmidt process
        y_axis_orthogonal = y_axis - np.dot(y_axis, x_axis_normalized) * x_axis_normalized
        y_axis_normalized = y_axis_orthogonal / np.linalg.norm(y_axis_orthogonal)
        # Z-axis is the cross product
        z_axis_normalized = np.cross(x_axis_normalized, y_axis_normalized)
        z_axis_normalized = z_axis_normalized / np.linalg.norm(z_axis_normalized)

        self.pallet_x_dimension = np.linalg.norm(x_axis)  # meters
        self.pallet_y_dimension = np.linalg.norm(y_axis)

        
        
        # Calculate scaling factors from optimizer units to meters
        # Optimizer uses a 4×2 pallet in its internal units
        self.optimizer_x_to_m = 1 / 100  # Assuming optimizer units are in cm
        self.optimizer_y_to_m = 1 / 100  # Assuming optimizer units are in cm
        self.optimizer_z_to_m = 1 / 100
        
        # Log the scaling factors
        self.get_logger().info(f'Optimizer to world scaling factors:')
        self.get_logger().info(f'  X: 1 optimizer unit = {self.optimizer_x_to_m:.4f} meters')
        self.get_logger().info(f'  Y: 1 optimizer unit = {self.optimizer_y_to_m:.4f} meters')
        self.get_logger().info(f'  Z: 1 optimizer unit = {self.optimizer_z_to_m:.4f} meters')
        
        # Create transformation matrix
        rotation_matrix = np.column_stack((x_axis_normalized, y_axis_normalized, z_axis_normalized))
        translation = top_left
        
        # Full transformation matrix
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = rotation_matrix
        self.transformation_matrix[:3, 3] = translation
        
        self.get_logger().info('Box Stack Service is ready with basic visualization')
        self.get_logger().info(f'Pallet dimensions: {self.pallet_x_dimension:.4f} x {self.pallet_y_dimension:.4f} meters')
        
        # Ensure an empty visualization is shown at startup to confirm visualization is working
        #self.optimizer.visualize_stack()

    def create_pallet_aligned_orientations(self, pallet_x_axis, pallet_y_axis, pallet_z_axis):
        """
        Create orientations where gripper is explicitly aligned with pallet coordinates:
        - Gripper's X-axis aligns with pallet's X-axis
        - Gripper's Y-axis aligns with pallet's Y-axis
        - Gripper's Z-axis points downward (opposite of pallet's Z-axis)
        
        Args:
            pallet_x_axis: Normalized X-axis of the pallet coordinate system
            pallet_y_axis: Normalized Y-axis of the pallet coordinate system
            pallet_z_axis: Normalized Z-axis of the pallet coordinate system
        
        Returns:
            List of quaternion options with different yaw rotations
        """
        # Normalize all input axes to be safe
        pallet_x_axis = pallet_x_axis / np.linalg.norm(pallet_x_axis)
        pallet_y_axis = pallet_y_axis / np.linalg.norm(pallet_y_axis)
        pallet_z_axis = pallet_z_axis / np.linalg.norm(pallet_z_axis)
        
        # 1. Set gripper X-axis to align with pallet X-axis
        gripper_x = pallet_x_axis.copy()
        
        # 2. Set gripper Y-axis to align with pallet Y-axis
        gripper_y = pallet_y_axis.copy()
        
        # 3. Set gripper Z-axis to point downward (opposite pallet Z-axis)
        gripper_z = -pallet_z_axis.copy()
        
        # Create rotation matrix from these axes
        rotation_matrix = np.column_stack((gripper_x, gripper_y, gripper_z))
        
        # Verify orthogonality (should be close to identity matrix)
        check = np.round(rotation_matrix.T @ rotation_matrix, 5)
        print("Orthogonality check:\n", check)
        
        # Convert to quaternion
        base_quat = mat2quat(rotation_matrix)
        
        # Generate 4 quaternion options with different yaw rotations
        yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
        quaternion_options = []
        
        # Define helper functions for quaternion operations
        def q_yaw(theta):
            return np.array([0.0, 0.0, np.sin(theta/2), np.cos(theta/2)])
        
        def quat_mul(q, r):
            x1, y1, z1, w1 = q
            x2, y2, z2, w2 = r
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])
        
        # Generate options with different yaw rotations
        num_q =1
        for θ in yaws:
            q = quat_mul(base_quat, q_yaw(θ))
            quaternion_options.append(q)
            self.get_logger().info(f"Quaternion option {num_q+1} with yaw {np.degrees(θ):.1f}°: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
            num_q += 1

        
        return quaternion_options

    def create_fixed_gripper_orientations(self):
        """
        Create gripper orientations with fixed orientation where:
        - z-axis is pointing downward (in world frame)
        - x-axis is aligned with pallet x-axis
        - y-axis completes the right-handed coordinate system
        """
        # Create a rotation matrix where:
        # 1. Gripper z-axis points along global -Z (downward)
        # 2. Gripper x-axis points along global X
        # 3. Gripper y-axis is determined by the right-hand rule
        
        # Define exact gripper axes in world frame
        gripper_z_in_world = np.array([0, 0, -1])  # Points downward
        gripper_x_in_world = np.array([1, 0, 0])   # Points in global X direction
        
        # Make sure x is perpendicular to z
        gripper_x_in_world = gripper_x_in_world - np.dot(gripper_x_in_world, gripper_z_in_world) * gripper_z_in_world
        gripper_x_in_world = gripper_x_in_world / np.linalg.norm(gripper_x_in_world)
        
        # Create y using the right-hand rule
        gripper_y_in_world = np.cross(gripper_z_in_world, gripper_x_in_world)
        
        # Create rotation matrix with explicit columns
        rotation_matrix = np.column_stack((gripper_x_in_world, gripper_y_in_world, gripper_z_in_world))
        
        # Convert to quaternion
        base_quaternion = mat2quat(rotation_matrix)
        
        # Generate 4 quaternion options with rotations around z-axis
        quaternion_options = []
        for yaw in [0, np.pi/2, np.pi, 3*np.pi/2]:
            # Create rotation matrix for just the yaw component
            yaw_matrix = euler2mat(0, 0, yaw, 'sxyz')
            # Combine base rotation with yaw rotation
            combined_rotation = np.dot(rotation_matrix, yaw_matrix)
            # Convert to quaternion
            quat = mat2quat(combined_rotation)
            quaternion_options.append(quat)
        
        return quaternion_options

    def create_explicit_gripper_orientations(self):
        """Alternative approach using explicit Euler angles"""
        quaternion_options = []
        
        # Create 4 quaternions with different yaw angles
        for yaw in [0, np.pi/2, np.pi, 3*np.pi/2]:
            # Roll = 180°, Pitch = 0°, Yaw = variable
            # This creates a gripper where z-axis points downward and x-axis varies with yaw
            rotation_matrix = euler2mat(np.pi, 0, yaw, 'sxyz')
            quat = mat2quat(rotation_matrix)
            quaternion_options.append(quat)
        
        return quaternion_options
    
    def create_gripper_orientations(self, x_axis_normalized, y_axis_normalized, z_axis_normalized):
        """
        Create gripper orientations where:
        - x-axis is aligned with the input x_axis
        - y-axis is opposite to the input y_axis
        - z-axis points downward (opposite to the input z_axis)
        
        Args:
            x_axis_normalized: Normalized x-axis from the original coordinate system
            y_axis_normalized: Normalized y-axis from the original coordinate system
            z_axis_normalized: Normalized z-axis from the original coordinate system
            
        Returns:
            List of quaternion options (4 orientations with different yaws)
        """
        # 1. Keep x-axis aligned with the current x-axis
        gripper_x = x_axis_normalized.copy()
        
        # 2. Make y-axis opposite of the current y-axis
        gripper_y = -y_axis_normalized.copy()
        
        # 3. Make z-axis point downward (opposite of current z-axis)
        gripper_z = -z_axis_normalized.copy()
        
        # Create rotation matrix from these orthonormal vectors
        gripper_rotation_matrix = np.column_stack((gripper_x, gripper_y, gripper_z))
        
        # Verify the matrix is orthogonal (should be close to identity)
        check_matrix = gripper_rotation_matrix.T @ gripper_rotation_matrix
        print("Orthogonality check:", check_matrix)  # Should be close to identity matrix
        
        # Convert to quaternion
        base_quaternion = mat2quat(gripper_rotation_matrix)
        print("Base gripper quaternion:", base_quaternion)
        
        # Generate 4 quaternion options by rotating around z-axis
        yaws = [0, np.pi/2, np.pi, 3*np.pi/2]
        quaternion_options = []
        
        for θ in yaws:
            q = self.quat_mul(base_quaternion, self.q_yaw(θ))
            quaternion_options.append(q)
            print(f"Quaternion with yaw {θ:.2f}° ({np.degrees(θ)}°):", q)
        
        return quaternion_options
    
    # Function to create quaternion for yaw rotation
    def q_yaw(self, theta):
        # Quaternion for rotation about local z-axis by θ:
        # q = [x, y, z, w] = [0, 0, sin(θ/2), cos(θ/2)]
        return np.array([0.0, 0.0, np.sin(theta/2), np.cos(theta/2)])

    # Function to multiply quaternions
    def quat_mul(self, q, r):
        x1, y1, z1, w1 = q
        x2, y2, z2, w2 = r
        # Hamilton product r ⊗ q (first q, then r)
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def publish_stack_state(self):
        """Publish the current state of the box stack"""
        from stack_optimization_interfaces.msg import BoxState, BoxInfo
        from std_msgs.msg import Header
        
        if self.optimizer.box_count == 0:
            return  # Don't publish empty states
        
        # Create message
        msg = BoxState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pallet"
        
        # Add box information
        for i, box in enumerate(self.optimizer.placed_boxes):
            x, y, z, w, l, h, quat = box  # x, y, z, width, length, height, quaternion
            
            box_info = BoxInfo()
            box_info.id = i
            box_info.x = float(x)  # Convert to float
            box_info.y = float(y)
            box_info.z = float(z)
            box_info.width = float(w)  # Now width corresponds to x
            box_info.length = float(l)  # Now length corresponds to y
            box_info.height = float(h)
            
            # Convert quaternion to ROS format
            box_info.orientation = Quaternion(
                x=float(quat[0]),
                y=float(quat[1]),
                z=float(quat[2]),
                w=float(quat[3])
            )
            
            msg.boxes.append(box_info)
        
        # Publish the message
        self.box_state_publisher.publish(msg)

    def visualize_stack(self):
        """Call the optimizer's visualization method if boxes exist"""
        if self.optimizer.box_count > 0:
            self.get_logger().info(f'Visualizing stack with {self.optimizer.box_count} boxes')
            self.optimizer.visualize_stack()
        else:
            self.get_logger().info('No boxes to visualize yet')
            self.optimizer.visualize_stack()  # Show empty pallet
    
    def transform_point(self, local_point):
        """
        Transform a point from local optimizer coordinates to global coordinates
        
        Args:
            local_point: Point in optimizer's internal units
            
        Returns:
            Point in global coordinates (meters)
        """
        # First, scale the optimizer units to meters
        # Apply different scaling factors for each dimension
        scaled_point = np.array([
            local_point[0] * self.optimizer_x_to_m,
            local_point[1] * self.optimizer_y_to_m,
            local_point[2] * self.optimizer_z_to_m
        ])
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(scaled_point, 1)
        
        # Apply transformation matrix to get global coordinates
        transformed_point = self.transformation_matrix @ point_homogeneous
        
        # Return 3D point in global coordinates (meters)
        return transformed_point[:3]
    
    def handle_box_stack_request(self, request, response):
        """
        Handle a box stacking request
        
        Note on units:
        - Input box dimensions (request) are in optimizer units
        - Internal optimizer coordinates are in abstract units
        - Output position (response) is in global coordinate system meters
        
        In this modified implementation, we assume:
        - x corresponds to width (not length as in previous code)
        - y corresponds to length (not width as in previous code)
        """
        self.get_logger().info(f'Received request to place box: {request.width}x{request.length}x{request.height} units')
        
        # Check if dimensions are valid
        if request.width <= 0 or request.length <= 0 or request.height <= 0:
            self.get_logger().error('Invalid box dimensions')
            response.success = False
            return response
        
        # Place the box (in optimizer's internal units)
        # Swap length and width when calling add_box to change the interpretation
        success, position, quat = self.optimizer.add_box(
            request.width,  # Previously was request.length
            request.length, # Previously was request.width
            request.height,
            request.change_stack_allowed
        )
        
        if success:
            # Set response
            response.success = True
            
            # Determine the actual dimensions of the box based on rotation (still in optimizer units)
            # Updated to match the new assumption that x is width, not length
            #Log width and length before swapping
            self.get_logger().info(f'Box dimensions before swapping (optimizer units): {request.width}x{request.length}x{request.height}')
            if quat[2] > 0:  # Rotated 90 degrees around z-axis
                box_width = request.length
                box_length = request.width
            else:
                box_width = request.width
                box_length = request.length
            #Log box dimensions after swapping
            self.get_logger().info(f'Box dimensions after swapping (optimizer units): ({box_width}, {box_length}, {request.height})')

            # Calculate center position from corner position in local coordinates (optimizer units)
            # Updated to match the new assumption that x is width, not length
            local_center_x = position[0] + box_width / 2
            local_center_y = position[1] + box_length / 2
            local_center_z = position[2] + request.height  #box needs to be placed on top of the stack
            
            # Convert to global coordinates (in meters) with proper scaling
            local_center = np.array([local_center_x, local_center_y, local_center_z])
            global_center = self.transform_point(local_center)
            
            # Set center position in response with global coordinates (in meters)
            response.position.x = float(global_center[0])
            response.position.y = float(global_center[1])
            response.position.z = float(global_center[2])
            
            # Set x_dimension and y_dimension in response (in optimizer units)
            # Updated to match the new assumption that x is width, not length
            response.x_dimension = int(box_width)  # Previously was box_length
            response.y_dimension = int(box_length) # Previously was box_width
            
            # Create quaternion for response
            # Note: The quaternion should also be transformed to align with the global coordinate system
            # For simplicity, we're keeping the original implementation
            q = Quaternion()
            q.x = float(quat[0])
            q.y = float(quat[1])
            q.z = float(quat[2])
            q.w = float(quat[3])
            
            # Append to orientations array
            response.orientations = [q]
            
            # Publish transform for visualization
            self.publish_box_tf(f'box_{self.optimizer.box_count - 1}', position, quat)
            
            # Explicitly visualize the updated stack - IMPORTANT for visibility
            #self.visualize_stack()
            
            self.publish_stack_state()
            
            # Log both in optimizer units and meters
            self.get_logger().info(f'Box placed at local center (optimizer units): ({local_center_x}, {local_center_y}, {local_center_z})')
            self.get_logger().info(f'Box placed at global center (meters): ({global_center[0]:.4f}, {global_center[1]:.4f}, {global_center[2]:.4f})')
            #Print quaternion list
            for i, q in enumerate(response.orientations):
                self.get_logger().info(f'Box Orientation {i}: [{q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}]')

            self.get_logger().info(f'Box dimensions (optimizer units): {response.x_dimension}x{response.y_dimension}')
            self.get_logger().info(f'Box dimensions (meters): {response.x_dimension * self.optimizer_x_to_m:.4f}x{response.y_dimension * self.optimizer_y_to_m:.4f}')
        else:
            response.success = False
            self.get_logger().warn('Failed to place box')
        
        return response
    
    def publish_box_tf(self, frame_id, position, quat):
        """Publish a transform for visualization"""
        # This would need to be implemented for ROS2 visualization
        # Typically using tf2_ros.TransformBroadcaster
        pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create and start the service
        box_stack_service = BoxStackService()
        

        # Main loop
        rclpy.spin(box_stack_service)
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        plt.close('all')
        if 'box_stack_service' in locals():
            box_stack_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
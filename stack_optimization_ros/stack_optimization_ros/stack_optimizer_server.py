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

class BoxStackOptimizer:
    def __init__(self, pallet_x=80, pallet_y=60, max_height=100, margin=5):
        self.pallet_x = pallet_x
        self.pallet_y = pallet_y
        self.max_height = max_height
        self.margin = margin
        
        # Track placed boxes
        self.placed_boxes = []  # (x, y, z, l, w, h, quaternion)
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
    
    def add_box(self, length, width, height, change_stack_allowed=False):
        """
        Add a new box to the stack
        
        Args:
            length, width, height: Box dimensions
            change_stack_allowed: Whether to allow rearranging existing boxes
            
        Returns:
            success: Whether placement succeeded
            position: (x, y, z) position for the box
            quaternion: Orientation of the box
        """
        if change_stack_allowed:
            # We can rearrange all boxes, including previously placed ones
            boxes = self.placed_boxes.copy()
            boxes.append((length, width, height))
            
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
                        box_length, box_width = box_dims[1], box_dims[0]  # Swap length/width
                    else:
                        quat = euler2quat(0, 0, 0)  # No rotation
                        box_length, box_width = box_dims[0], box_dims[1]
                    
                    self.placed_boxes.append((x, y, z, box_length, box_width, box_dims[2], quat))
                
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
            success, position, rotation = self._optimize_new_box_placement(length, width, height)
            
            if success:
                # Create quaternion from rotation
                if rotation:
                    quat = euler2quat(0, 0, np.pi/2)
                    box_length, box_width = width, length  # Swap length/width
                else:
                    quat = euler2quat(0, 0, 0)  # No rotation
                    box_length, box_width = length, width
                
                self.placed_boxes.append((position[0], position[1], position[2], 
                                         box_length, box_width, height, quat))
                self.box_count += 1
                return True, position, quat
            else:
                return False, None, None
    
    def _optimize_stack(self, boxes):
        """Optimize the stacking of all boxes from scratch"""
        n = len(boxes)
        model = cp_model.CpModel()
        
        x, y, z = [], [], []
        r = []  # rotation variable
        length_vars, width_vars = [], []
        
        for i in range(n):
            l, w, h = boxes[i]
            # Add margin to x and y lower bounds
            x_i = model.NewIntVar(self.margin, self.pallet_x - self.margin, f'x_{i}')
            y_i = model.NewIntVar(self.margin, self.pallet_y - self.margin, f'y_{i}')
            z_i = model.NewIntVar(0, self.max_height, f'z_{i}')
            r_i = model.NewBoolVar(f'r_{i}')  # 0 = no rotation, 1 = 90 deg rotation
            
            l_i = model.NewIntVar(0, self.pallet_x, f'l_{i}')
            w_i = model.NewIntVar(0, self.pallet_y, f'w_{i}')
            model.Add(l_i == l).OnlyEnforceIf(r_i.Not())
            model.Add(l_i == w).OnlyEnforceIf(r_i)
            model.Add(w_i == w).OnlyEnforceIf(r_i.Not())
            model.Add(w_i == l).OnlyEnforceIf(r_i)
            
            # Make sure box fits on pallet with margin
            model.Add(x_i + l_i <= self.pallet_x - self.margin)
            model.Add(y_i + w_i <= self.pallet_y - self.margin)
            
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
            r.append(r_i)
            length_vars.append(l_i)
            width_vars.append(w_i)
        
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
                model.Add(x[i] + length_vars[i] + self.margin <= x[j]).OnlyEnforceIf(b1)
                model.Add(x[i] + length_vars[i] + self.margin > x[j]).OnlyEnforceIf(b1.Not())
                
                model.Add(x[j] + length_vars[j] + self.margin <= x[i]).OnlyEnforceIf(b2)
                model.Add(x[j] + length_vars[j] + self.margin > x[i]).OnlyEnforceIf(b2.Not())
                
                model.Add(y[i] + width_vars[i] + self.margin <= y[j]).OnlyEnforceIf(b3)
                model.Add(y[i] + width_vars[i] + self.margin > y[j]).OnlyEnforceIf(b3.Not())
                
                model.Add(y[j] + width_vars[j] + self.margin <= y[i]).OnlyEnforceIf(b4)
                model.Add(y[j] + width_vars[j] + self.margin > y[i]).OnlyEnforceIf(b4.Not())
                
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
                model.AddMinEquality(x_overlap_end, [x[i] + length_vars[i], x[j] + length_vars[j]])
                
                # Compute overlap in y-direction
                model.AddMaxEquality(y_overlap_start, [y[i], y[j]])
                model.AddMinEquality(y_overlap_end, [y[i] + width_vars[i], y[j] + width_vars[j]])
                
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
                # This ensures the box isn't just barely touching its support
                min_overlap_x = model.NewIntVar(0, self.pallet_x, f'min_overlap_x_{i}_{j}')
                # Using multiplication instead of division for IntVar
                model.Add(2 * min_overlap_x == length_vars[i])
                
                min_overlap_y = model.NewIntVar(0, self.pallet_y, f'min_overlap_y_{i}_{j}')
                # Using multiplication instead of division for IntVar
                model.Add(2 * min_overlap_y == width_vars[i])
                
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
        
        # Objective: minimize total height
        max_z = model.NewIntVar(0, self.max_height, 'max_z')
        for i in range(n):
            model.Add(max_z >= z[i] + boxes[i][2])
        model.Minimize(max_z)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # Limit solving time
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
    
    def _optimize_new_box_placement(self, length, width, height):
        """Place a new box on the existing stack"""
        model = cp_model.CpModel()
        
        # Variables for the new box
        # Add margin to x and y boundaries
        x_new = model.NewIntVar(self.margin, self.pallet_x - self.margin, 'x_new')
        y_new = model.NewIntVar(self.margin, self.pallet_y - self.margin, 'y_new')
        z_new = model.NewIntVar(0, self.max_height, 'z_new')
        r_new = model.NewBoolVar('r_new')  # 0 = no rotation, 1 = 90 deg rotation
        
        l_new = model.NewIntVar(0, self.pallet_x, 'l_new')
        w_new = model.NewIntVar(0, self.pallet_y, 'w_new')
        model.Add(l_new == length).OnlyEnforceIf(r_new.Not())
        model.Add(l_new == width).OnlyEnforceIf(r_new)
        model.Add(w_new == width).OnlyEnforceIf(r_new.Not())
        model.Add(w_new == length).OnlyEnforceIf(r_new)
        
        # Make sure box fits on pallet with margin
        model.Add(x_new + l_new <= self.pallet_x - self.margin)
        model.Add(y_new + w_new <= self.pallet_y - self.margin)
        
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
            model.Add(x_new + l_new + self.margin <= x_i).OnlyEnforceIf(b1)
            model.Add(x_new + l_new + self.margin > x_i).OnlyEnforceIf(b1.Not())
            
            model.Add(x_i + l_i + self.margin <= x_new).OnlyEnforceIf(b2)
            model.Add(x_i + l_i + self.margin > x_new).OnlyEnforceIf(b2.Not())
            
            model.Add(y_new + w_new + self.margin <= y_i).OnlyEnforceIf(b3)
            model.Add(y_new + w_new + self.margin > y_i).OnlyEnforceIf(b3.Not())
            
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
            model.AddMinEquality(x_overlap_end, [x_new + l_new, x_i + l_i])
            
            # Compute overlap in y-direction
            model.AddMaxEquality(y_overlap_start, [y_new, y_i])
            model.AddMinEquality(y_overlap_end, [y_new + w_new, y_i + w_i])
            
            # Calculate overlap dimensions
            x_overlap = model.NewIntVar(-self.pallet_x, self.pallet_x, f'x_overlap_new_{i}')
            y_overlap = model.NewIntVar(-self.pallet_y, self.pallet_y, f'y_overlap_new_{i}')
            model.Add(x_overlap == x_overlap_end - x_overlap_start)
            model.Add(y_overlap == y_overlap_end - y_overlap_start)
            
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
            model.Add(2 * min_overlap_x == l_new)
            
            min_overlap_y = model.NewIntVar(0, self.pallet_y, f'min_overlap_y_new_{i}')
            # Using multiplication instead of division for IntVar
            model.Add(2 * min_overlap_y == w_new)
            
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
        
        # Objective: place the box at the lowest possible height
        model.Minimize(z_new)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # Limit solving time
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return True, (solver.Value(x_new), solver.Value(y_new), solver.Value(z_new)), solver.Value(r_new)
        else:
            return False, None, None
    
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
        self.ax.set_xlim(0, self.pallet_x)
        self.ax.set_ylim(0, self.pallet_y)
        self.ax.set_zlim(0, self.max_height)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
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
            x, y, z, l, w, h, _ = box

            #Log dimensions and properties
            print(f'Box {i}: Position ({x}, {y}, {z}), Size ({l}, {w}, {h})')
            
            # Plot the box
            self._plot_box(self.ax, (x, y, z), (l, w, h), cmap(i))
            
            # Plot margins if enabled
            if self.show_margins:
                margin_origin = (x - self.margin, y - self.margin, z)
                margin_size = (l + 2*self.margin, w + 2*self.margin, h)
                self._plot_margin(self.ax, margin_origin, margin_size, cmap(i))
        
        # Add view controls
        self.ax.view_init(elev=30, azim=45)  # Default view
        
        # Show the figure with blocking=True to ensure it stays visible
        plt.show(block=False)
    
    def _plot_box(self, ax, origin, size, color):
        """Helper function to plot a box"""
        x0, y0, z0 = origin
        dx, dy, dz = size
        
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
        """Helper function to plot the safety margin boundaries"""
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
        self.optimizer = BoxStackOptimizer(margin=5)  # Use 5 units as requested
        
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
        top_left = np.array([0.5792, 0.4032, -0.8384])
        top_right = np.array([1.1630, 0.4828, -0.8489])
        bottom_left = np.array([0.6782, -0.3763, -0.8343])
        
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
        
        # Store pallet dimensions in meters
        self.pallet_x_dimension = np.linalg.norm(x_axis)  # meters
        self.pallet_y_dimension = np.linalg.norm(y_axis)  # meters
        
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
            x, y, z, l, w, h, quat = box
            
            box_info = BoxInfo()
            box_info.id = i
            box_info.x = float(x)  # Convert to float
            box_info.y = float(y)
            box_info.z = float(z)
            box_info.length = float(l)
            box_info.width = float(w)
            box_info.height = float(h)
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
        """
        self.get_logger().info(f'Received request to place box: {request.width}x{request.length}x{request.height} units')
        
        # Check if dimensions are valid
        if request.width <= 0 or request.length <= 0 or request.height <= 0:
            self.get_logger().error('Invalid box dimensions')
            response.success = False
            return response
        
        # Place the box (in optimizer's internal units)
        success, position, quat = self.optimizer.add_box(
            request.length, 
            request.width, 
            request.height,
            request.change_stack_allowed
        )
        
        if success:
            # Set response
            response.success = True
            
            # Determine the actual dimensions of the box based on rotation (still in optimizer units)
            if quat[2] > 0:  # Rotated 90 degrees around z-axis
                box_length = request.width
                box_width = request.length
            else:
                box_length = request.length
                box_width = request.width
            
            # Calculate center position from corner position in local coordinates (optimizer units)
            local_center_x = position[0] + box_length / 2
            local_center_y = position[1] + box_width / 2
            local_center_z = position[2] + request.height  #box needs to be placed on top of the stack
            
            # Convert to global coordinates (in meters) with proper scaling
            local_center = np.array([local_center_x, local_center_y, local_center_z])
            global_center = self.transform_point(local_center)
            
            # Set center position in response with global coordinates (in meters)
            response.position.x = float(global_center[0])
            response.position.y = float(global_center[1])
            response.position.z = float(global_center[2])
            
            # Set x_dimension and y_dimension in response (in optimizer units)
            # Note: If client needs these in meters, multiply by the appropriate scaling factor
            response.x_dimension = int(box_length)
            response.y_dimension = int(box_width)
            
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
            self.visualize_stack()
            
            self.publish_stack_state()
            
            # Log both in optimizer units and meters
            self.get_logger().info(f'Box placed at local center (optimizer units): ({local_center_x}, {local_center_y}, {local_center_z})')
            self.get_logger().info(f'Box placed at global center (meters): ({global_center[0]:.4f}, {global_center[1]:.4f}, {global_center[2]:.4f})')
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
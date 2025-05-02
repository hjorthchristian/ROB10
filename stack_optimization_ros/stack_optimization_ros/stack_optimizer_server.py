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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
plt.ion()

class BoxStackOptimizer:
    def __init__(self, pallet_x=4, pallet_y=2, max_height=10):
        self.pallet_x = pallet_x
        self.pallet_y = pallet_y
        self.max_height = max_height
        
        # Track placed boxes
        self.placed_boxes = []  # (x, y, z, l, w, h, quaternion)
        self.box_count = 0
    
    def reset(self):
        """Reset the optimizer state"""
        self.placed_boxes = []
        self.box_count = 0
    
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
            x_i = model.NewIntVar(0, self.pallet_x, f'x_{i}')
            y_i = model.NewIntVar(0, self.pallet_y, f'y_{i}')
            z_i = model.NewIntVar(0, self.max_height, f'z_{i}')
            r_i = model.NewBoolVar(f'r_{i}')  # 0 = no rotation, 1 = 90 deg rotation
            
            l_i = model.NewIntVar(0, self.pallet_x, f'l_{i}')
            w_i = model.NewIntVar(0, self.pallet_y, f'w_{i}')
            model.Add(l_i == l).OnlyEnforceIf(r_i.Not())
            model.Add(l_i == w).OnlyEnforceIf(r_i)
            model.Add(w_i == w).OnlyEnforceIf(r_i.Not())
            model.Add(w_i == l).OnlyEnforceIf(r_i)
            
            model.Add(x_i + l_i <= self.pallet_x)
            model.Add(y_i + w_i <= self.pallet_y)
            
            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
            r.append(r_i)
            length_vars.append(l_i)
            width_vars.append(w_i)
        
        # No-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                b1 = model.NewBoolVar(f'no_overlap_x1_{i}_{j}')
                b2 = model.NewBoolVar(f'no_overlap_x2_{i}_{j}')
                b3 = model.NewBoolVar(f'no_overlap_y1_{i}_{j}')
                b4 = model.NewBoolVar(f'no_overlap_y2_{i}_{j}')
                b5 = model.NewBoolVar(f'no_overlap_z1_{i}_{j}')
                b6 = model.NewBoolVar(f'no_overlap_z2_{i}_{j}')
                
                model.Add(x[i] + length_vars[i] <= x[j]).OnlyEnforceIf(b1)
                model.Add(x[i] + length_vars[i] > x[j]).OnlyEnforceIf(b1.Not())
                
                model.Add(x[j] + length_vars[j] <= x[i]).OnlyEnforceIf(b2)
                model.Add(x[j] + length_vars[j] > x[i]).OnlyEnforceIf(b2.Not())
                
                model.Add(y[i] + width_vars[i] <= y[j]).OnlyEnforceIf(b3)
                model.Add(y[i] + width_vars[i] > y[j]).OnlyEnforceIf(b3.Not())
                
                model.Add(y[j] + width_vars[j] <= y[i]).OnlyEnforceIf(b4)
                model.Add(y[j] + width_vars[j] > y[i]).OnlyEnforceIf(b4.Not())
                
                model.Add(z[i] + boxes[i][2] <= z[j]).OnlyEnforceIf(b5)
                model.Add(z[i] + boxes[i][2] > z[j]).OnlyEnforceIf(b5.Not())
                
                model.Add(z[j] + boxes[j][2] <= z[i]).OnlyEnforceIf(b6)
                model.Add(z[j] + boxes[j][2] > z[i]).OnlyEnforceIf(b6.Not())
                
                model.AddBoolOr([b1, b2, b3, b4, b5, b6])
        
        # Stability constraints
        for i in range(n):
            support_conditions = []
            
            # Support on pallet
            support_on_pallet = model.NewBoolVar(f'support_pallet_{i}')
            model.Add(z[i] == 0).OnlyEnforceIf(support_on_pallet)
            model.Add(z[i] != 0).OnlyEnforceIf(support_on_pallet.Not())
            support_conditions.append(support_on_pallet)
            
            for j in range(n):
                if i == j:
                    continue
                
                support_x = model.NewBoolVar(f'support_x_{i}_{j}')
                support_y = model.NewBoolVar(f'support_y_{i}_{j}')
                support_z = model.NewBoolVar(f'support_z_{i}_{j}')
                support_all = model.NewBoolVar(f'supported_{i}_{j}')
                
                model.Add(x[i] >= x[j]).OnlyEnforceIf(support_x)
                model.Add(x[i] + length_vars[i] <= x[j] + length_vars[j]).OnlyEnforceIf(support_x)
                model.Add(y[i] >= y[j]).OnlyEnforceIf(support_y)
                model.Add(y[i] + width_vars[i] <= y[j] + width_vars[j]).OnlyEnforceIf(support_y)
                model.Add(z[i] == z[j] + boxes[j][2]).OnlyEnforceIf(support_z)
                
                model.AddBoolAnd([support_x, support_y, support_z]).OnlyEnforceIf(support_all)
                model.AddBoolOr([support_x.Not(), support_y.Not(), support_z.Not()]).OnlyEnforceIf(support_all.Not())
                
                support_conditions.append(support_all)
            
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
        x_new = model.NewIntVar(0, self.pallet_x, 'x_new')
        y_new = model.NewIntVar(0, self.pallet_y, 'y_new')
        z_new = model.NewIntVar(0, self.max_height, 'z_new')
        r_new = model.NewBoolVar('r_new')  # 0 = no rotation, 1 = 90 deg rotation
        
        l_new = model.NewIntVar(0, self.pallet_x, 'l_new')
        w_new = model.NewIntVar(0, self.pallet_y, 'w_new')
        model.Add(l_new == length).OnlyEnforceIf(r_new.Not())
        model.Add(l_new == width).OnlyEnforceIf(r_new)
        model.Add(w_new == width).OnlyEnforceIf(r_new.Not())
        model.Add(w_new == length).OnlyEnforceIf(r_new)
        
        model.Add(x_new + l_new <= self.pallet_x)
        model.Add(y_new + w_new <= self.pallet_y)
        
        # No-overlap constraints with existing boxes
        for i, box in enumerate(self.placed_boxes):
            x_i, y_i, z_i, l_i, w_i, h_i, _ = box
            
            b1 = model.NewBoolVar(f'no_overlap_x1_new_{i}')
            b2 = model.NewBoolVar(f'no_overlap_x2_new_{i}')
            b3 = model.NewBoolVar(f'no_overlap_y1_new_{i}')
            b4 = model.NewBoolVar(f'no_overlap_y2_new_{i}')
            b5 = model.NewBoolVar(f'no_overlap_z1_new_{i}')
            b6 = model.NewBoolVar(f'no_overlap_z2_new_{i}')
            
            model.Add(x_new + l_new <= x_i).OnlyEnforceIf(b1)
            model.Add(x_new + l_new > x_i).OnlyEnforceIf(b1.Not())
            
            model.Add(x_i + l_i <= x_new).OnlyEnforceIf(b2)
            model.Add(x_i + l_i > x_new).OnlyEnforceIf(b2.Not())
            
            model.Add(y_new + w_new <= y_i).OnlyEnforceIf(b3)
            model.Add(y_new + w_new > y_i).OnlyEnforceIf(b3.Not())
            
            model.Add(y_i + w_i <= y_new).OnlyEnforceIf(b4)
            model.Add(y_i + w_i > y_new).OnlyEnforceIf(b4.Not())
            
            model.Add(z_new + height <= z_i).OnlyEnforceIf(b5)
            model.Add(z_new + height > z_i).OnlyEnforceIf(b5.Not())
            
            model.Add(z_i + h_i <= z_new).OnlyEnforceIf(b6)
            model.Add(z_i + h_i > z_new).OnlyEnforceIf(b6.Not())
            
            model.AddBoolOr([b1, b2, b3, b4, b5, b6])
        
        # Stability constraints
        support_conditions = []
        
        # Support on pallet
        support_on_pallet = model.NewBoolVar('support_pallet_new')
        model.Add(z_new == 0).OnlyEnforceIf(support_on_pallet)
        model.Add(z_new != 0).OnlyEnforceIf(support_on_pallet.Not())
        support_conditions.append(support_on_pallet)
        
        # Support on existing boxes
        for i, box in enumerate(self.placed_boxes):
            x_i, y_i, z_i, l_i, w_i, h_i, _ = box
            
            support_x = model.NewBoolVar(f'support_x_new_{i}')
            support_y = model.NewBoolVar(f'support_y_new_{i}')
            support_z = model.NewBoolVar(f'support_z_new_{i}')
            support_all = model.NewBoolVar(f'supported_new_{i}')
            
            model.Add(x_new >= x_i).OnlyEnforceIf(support_x)
            model.Add(x_new + l_new <= x_i + l_i).OnlyEnforceIf(support_x)
            model.Add(y_new >= y_i).OnlyEnforceIf(support_y)
            model.Add(y_new + w_new <= y_i + w_i).OnlyEnforceIf(support_y)
            model.Add(z_new == z_i + h_i).OnlyEnforceIf(support_z)
            
            model.AddBoolAnd([support_x, support_y, support_z]).OnlyEnforceIf(support_all)
            model.AddBoolOr([support_x.Not(), support_y.Not(), support_z.Not()]).OnlyEnforceIf(support_all.Not())
            
            support_conditions.append(support_all)
        
        # Require the box to be supported
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
    
    def visualize_stack(self):
        """Visualize the current stack of boxes"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, max(self.pallet_x, self.pallet_y))
        ax.set_ylim(0, max(self.pallet_x, self.pallet_y))
        ax.set_zlim(0, self.max_height)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Box Stack")
        
        if not self.placed_boxes:
            plt.show(block=False)
            return
        
        cmap = cm.get_cmap('tab20', len(self.placed_boxes))
        
        for i, box in enumerate(self.placed_boxes):
            x, y, z, l, w, h, _ = box
            origin = (x, y, z)
            size = (l, w, h)
            self._plot_box(ax, origin, size, cmap(i))
        
        # Draw pallet base
        ax.bar3d(0, 0, 0, self.pallet_x, self.pallet_y, 0.1, color='gray', alpha=0.3, shade=True)
        
        # Use non-blocking mode
        plt.show(block=False)
        plt.pause(0.001)  # Small pause to allow the GUI to update
    
    def _plot_box(self, ax, origin, size, color):
        """Helper function to plot a box"""
        x0, y0, z0 = origin
        dx, dy, dz = size
        
        # Vertices
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
        
        # Faces
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],  # bottom
            [verts[4], verts[5], verts[6], verts[7]],  # top
            [verts[0], verts[1], verts[5], verts[4]],  # front
            [verts[1], verts[2], verts[6], verts[5]],  # right
            [verts[2], verts[3], verts[7], verts[6]],  # back
            [verts[3], verts[0], verts[4], verts[7]],  # left
        ]
        
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=0.8))


class BoxStackService(Node):
    def __init__(self):
        super().__init__('box_stack_optimizer_service')
        
        # Create the optimizer
        self.optimizer = BoxStackOptimizer()
        
        # Create the service
        self.srv = self.create_service(
            StackOptimizer, 
            'box_stack_optimizer', 
            self.handle_box_stack_request
        )
        
        # TF broadcaster for visualization
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create a visualization timer (update every 5 seconds)
        self.viz_timer = self.create_timer(5.0, self.visualize_stack)
        
        self.get_logger().info('Box Stack Service is ready')
    
    def visualize_stack(self):
        """Call the optimizer's visualization method if boxes exist"""
        if self.optimizer.box_count > 0:
            self.get_logger().info(f'Visualizing stack with {self.optimizer.box_count} boxes')
            self.optimizer.visualize_stack()
    
    def handle_box_stack_request(self, request, response):
        """Handle a box stacking request"""
        self.get_logger().info(f'Received request to place box: {request.width}x{request.length}x{request.height}')
        
        # Check if dimensions are valid
        if request.width <= 0 or request.length <= 0 or request.height <= 0:
            self.get_logger().error('Invalid box dimensions')
            response.success = False
            return response
        
        # Place the box
        success, position, quat = self.optimizer.add_box(
            request.length, 
            request.width, 
            request.height,
            request.change_stack_allowed
        )
        
        if success:
            # Set response
            response.success = True
            
            # Determine the actual dimensions of the box based on rotation
            if quat[2] > 0:  # Rotated 90 degrees around z-axis
                box_length = request.width
                box_width = request.length
            else:
                box_length = request.length
                box_width = request.width
            
            # Calculate center position from corner position
            center_x = position[0] + box_length / 2
            center_y = position[1] + box_width / 2
            center_z = position[2] + request.height / 2
            
            # Set center position in response
            response.position.x = float(center_x)
            response.position.y = float(center_y)
            response.position.z = float(center_z)
            
            # Set x_dimension and y_dimension in response
            response.x_dimension = int(box_length)
            response.y_dimension = int(box_width)
            
            # Create quaternion for response
            q = Quaternion()
            q.x = float(quat[0])
            q.y = float(quat[1])
            q.z = float(quat[2])
            q.w = float(quat[3])
            
            # Append to orientations array
            response.orientations = [q]
            
            # Publish transform for visualization
            self.publish_box_tf(f'box_{self.optimizer.box_count - 1}', position, quat)
            
            # Visualize the updated stack
            self.visualize_stack()
            
            self.get_logger().info(f'Box placed at center: ({center_x}, {center_y}, {center_z}), dimensions: {response.x_dimension}x{response.y_dimension}')
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
    box_stack_service = BoxStackService()
    
    try:
        rclpy.spin(box_stack_service)
    except KeyboardInterrupt:
        pass
    finally:
        box_stack_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
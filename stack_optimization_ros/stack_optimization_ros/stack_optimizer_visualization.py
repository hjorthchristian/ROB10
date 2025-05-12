#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from stack_optimization_interfaces.msg import BoxState
import threading
import matplotlib
matplotlib.use('TkAgg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from transforms3d.quaternions import quat2mat
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class BoxStackVisualizer(Node):
    def __init__(self):
        super().__init__('box_stack_visualizer')
        
        # Initialize state
        self.boxes = []
        self.data_lock = threading.Lock()
        self.update_needed = False
        self.first_draw = True
        
        # Initialize collections for updating instead of redrawing
        self.box_collections = {}
        self.box_labels = {}
        self.pallet = None
        
        # View state tracking
        self.azim = 45
        self.elev = 30
        
        # Initialize scaling boundaries - match server default values
        self.x_max = 80.0
        self.y_max = 60.0
        self.z_max = 100.0
        
        # Create color map for consistent box colors by ID
        self.colors = self._create_color_map(50)  # Support up to 50 distinct box IDs
        
        # Create subscription to box state
        self.subscription = self.create_subscription(
            BoxState, 'box_stack_state', self.box_state_callback, 10)
            
        # Initialize plot
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        
        # Connect mouse events to save view state
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Animation to update plot
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, save_count=100)
        
        self.get_logger().info('Box stack visualizer started with view preservation')
    
    def on_mouse_move(self, event):
        """Capture the current view angles when user interacts with plot"""
        if event.inaxes == self.ax:
            self.azim = self.ax.azim
            self.elev = self.ax.elev
    
    def _create_color_map(self, num_colors):
        """Create a fixed color map for box IDs"""
        # Use the same categorical colors as server for consistency
        base_colors = list(mcolors.TABLEAU_COLORS.values())
        base_colors.extend(list(mcolors.CSS4_COLORS.values())[:30])
        
        # Ensure we have enough colors
        while len(base_colors) < num_colors:
            base_colors.extend(base_colors[:num_colors - len(base_colors)])
            
        return base_colors[:num_colors]
    
    def get_box_color(self, box_id):
        """Get a consistent color for a specific box ID"""
        return self.colors[box_id % len(self.colors)]
    
    def box_state_callback(self, msg):
        """Callback for box state messages"""
        with self.data_lock:
            self.boxes = msg.boxes
            self.update_needed = True
            
            # Calculate current max dimensions with extra buffer
            if self.boxes:
                # For accurate visualization boundaries, consider the full box extents
                # Note: box position is the center, so we need to add half width/length
                max_x = max([box.x + box.length/2 for box in self.boxes]) * 1.2
                max_y = max([box.y + box.width/2 for box in self.boxes]) * 1.2
                max_z = max([box.z for box in self.boxes]) * 1.2  # z is top of box
                
                # Update max values only if they increase significantly
                if max_x > self.x_max:
                    self.x_max = max_x
                if max_y > self.y_max:
                    self.y_max = max_y
                if max_z > self.z_max:
                    self.z_max = max_z

    def plot_box(self, box):
        """Visualize a single box in 3D with proper center-to-corner conversion"""
        # Box dimensions (original)
        width = box.width
        length = box.length
        height = box.height
        
        # Log original values and quaternion
        orig_quat = [box.orientation.w, box.orientation.x, box.orientation.y, box.orientation.z]
        self.get_logger().info(f"Original quaternion: w={orig_quat[0]}, x={orig_quat[1]}, y={orig_quat[2]}, z={orig_quat[3]}")
        self.get_logger().info(f"Original dimensions: length={length}, width={width}")
        
        # CRITICAL FIX: Convert center position to corner position
        center_x = box.x
        center_y = box.y
        center_z = box.z
        
        # Calculate corner coordinates
        x = center_x - length/2
        y = center_y - width/2
        z = center_z - height  # The Z position is the top of the box, subtract height for bottom
        
        self.get_logger().info(f"Converting from center ({center_x}, {center_y}, {center_z}) to corner ({x}, {y}, {z})")
        
        # Define the vertices of the box relative to the corner
        verts = [
            [x, y, z],
            [x + length, y, z],
            [x + length, y + width, z],
            [x, y + width, z],
            [x, y, z + height],
            [x + length, y, z + height],
            [x + length, y + width, z + height],
            [x, y + width, z + height],
        ]
        
        # Check if we need to apply the quaternion rotation
        quat = [box.orientation.w, box.orientation.x, box.orientation.y, box.orientation.z]
        
        # Apply rotation if needed
        # Fix X-axis rotation - use identity rotation instead
        # The quaternion [0.7071, 0, 0, 0.7071] is a 90° X-axis rotation, but we want to avoid this
        x_rotation_quat = np.isclose(quat[0], 0.7071, atol=0.01) and np.isclose(quat[1], 0.7071, atol=0.01)
        
        if x_rotation_quat:
            # This is an X-axis rotation which is causing the boxes to be flipped 
            # Instead of using this quaternion, use identity orientation for visualization
            self.get_logger().info(f"Replacing problematic X-axis rotation with identity quaternion for box {box.id}")
            rotation_matrix = np.eye(3)  # Identity matrix - no rotation
        else:
            # If it's a different rotation, convert to rotation matrix
            self.get_logger().info(f"Applying rotation: w={quat[0]}, x={quat[1]}, y={quat[2]}, z={quat[3]}")
            rotation_matrix = quat2mat(quat)
        
        self.get_logger().info(f"Rotation matrix:\n{rotation_matrix}")
        
        # Calculate box center for rotation
        center = np.array([
            x + length/2,
            y + width/2,
            z + height/2
        ])
        
        # Apply rotation around center
        rotated_verts = []
        for v in verts:
            v_array = np.array(v)
            v_centered = v_array - center
            v_rotated = np.dot(rotation_matrix, v_centered)
            v_final = v_rotated + center
            rotated_verts.append(v_final.tolist())
        
        # Log the final vertices
        self.get_logger().info(f"Final vertices for box {box.id}:")
        for i, v in enumerate(rotated_verts):
            self.get_logger().info(f"  Vertex {i}: ({v[0]}, {v[1]}, {v[2]})")
        
        # Define faces using the standard order
        faces = [
            [rotated_verts[0], rotated_verts[1], rotated_verts[2], rotated_verts[3]],  # bottom
            [rotated_verts[4], rotated_verts[5], rotated_verts[6], rotated_verts[7]],  # top
            [rotated_verts[0], rotated_verts[1], rotated_verts[5], rotated_verts[4]],  # front
            [rotated_verts[1], rotated_verts[2], rotated_verts[6], rotated_verts[5]],  # right
            [rotated_verts[2], rotated_verts[3], rotated_verts[7], rotated_verts[6]],  # back
            [rotated_verts[3], rotated_verts[0], rotated_verts[4], rotated_verts[7]],  # left
        ]
        
        # Get consistent color for this box ID and plot
        box_color = self.get_box_color(box.id)
        collection = Poly3DCollection(faces, facecolors=box_color, edgecolors='k', alpha=0.7)
        self.ax.add_collection3d(collection)
        
        # Add box ID label
        label_x = center_x
        label_y = center_y
        label_z = center_z + 2  # Slightly above box
        label = self.ax.text(label_x, label_y, label_z, f"ID: {box.id}", 
                        color='black', backgroundcolor=box_color, 
                        fontweight='bold', fontsize=10)
        
        return collection, label
    def setup_plot(self):
        """Setup the 3D plot with labels and limits"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Box Stack Visualization')
        
        # Set consistent limits based on tracked max values
        self.ax.set_xlim([0, self.x_max])
        self.ax.set_ylim([0, self.y_max])
        self.ax.set_zlim([0, self.z_max])
        
        # Equal aspect ratio for better visualization
        # The default mplot3d equal aspect ratio can look distorted
        # Setting approximately equal aspect with fixed ratios
        x_range = self.x_max
        y_range = self.y_max
        z_range = self.z_max
        max_range = max(x_range, y_range, z_range)
        
        self.ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
        
        # Add grid for better depth perception
        self.ax.grid(True)
        
        # Set initial viewing angle (only if first time)
        if self.first_draw:
            self.ax.view_init(elev=30, azim=45)  # This angle worked well in the server
            self.first_draw = False
        else:
            # Restore previous view
            self.ax.view_init(elev=self.elev, azim=self.azim)

    def update_plot(self, frame):
        """Update plot with enhanced error handling and debugging"""
        with self.data_lock:
            if self.update_needed:
                try:
                    # Debug log before update
                    self.get_logger().info(f"Updating plot with {len(self.boxes)} boxes")
                    
                    # Save current view angles before clearing
                    if hasattr(self.ax, 'azim') and hasattr(self.ax, 'elev'):
                        self.azim = self.ax.azim
                        self.elev = self.ax.elev
                    
                    # Clear existing collections to redraw with new data
                    self.ax.clear()
                    self.setup_plot()
                    
                    # Draw a wireframe plane at z=0 to represent the pallet
                    pallet_x = np.linspace(0, self.x_max, 2)
                    pallet_y = np.linspace(0, self.y_max, 2)
                    X, Y = np.meshgrid(pallet_x, pallet_y)
                    Z = np.zeros_like(X)
                    self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)
                    
                    # Debug log boxes
                    self.get_logger().info(f"Preparing to plot {len(self.boxes)} boxes")
                    
                    # Plot boxes from received state
                    for box in self.boxes:
                        self.plot_box(box)
                    
                    # Add a legend showing box IDs and their colors
                    if self.boxes:
                        legend_handles = []
                        for box in self.boxes:
                            color = self.get_box_color(box.id)
                            legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=f"Box {box.id}"))
                        self.ax.legend(handles=legend_handles, loc='upper right')
                    
                    # Restore view angle
                    self.ax.view_init(elev=self.elev, azim=self.azim)
                    self.update_needed = False
                    
                    # Debug log after update
                    self.get_logger().info("Plot updated successfully")
                except Exception as e:
                    self.get_logger().error(f"Error updating plot: {str(e)}")
                    # Try to recover by setting update_needed to False
                    self.update_needed = False

def main(args=None):
    rclpy.init(args=args)
    visualizer = BoxStackVisualizer()
    
    try:
        # Create a separate thread for ROS spinning
        spin_thread = threading.Thread(target=rclpy.spin, args=(visualizer,))
        spin_thread.daemon = True
        spin_thread.start()
        
        # Start matplotlib main loop on the main thread
        plt.show()
        
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
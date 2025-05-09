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
        """Visualize a single box in 3D with proper quaternion orientation application"""
        # Box dimensions
        width = box.width
        length = box.length
        height = box.height
        #if box height not equal to z size then go up or som,ething like that gfor ground boxes
        # Get box position (server reports position as corner, not center)
        x = box.x
        y = box.y
        z = box.z   # Adjust if server reports z at top of box
        # Important: Override the quaternion to keep boxes upright
    # This cancels the rotation that's laying the boxes on their sides
        orig_quat = [box.orientation.w, box.orientation.x, box.orientation.y, box.orientation.z]
        
        # Check if this is the problematic X-axis rotation quaternion (approximately [0.7071, 0.7071, 0, 0])
        if np.isclose(orig_quat[0], 0.7071, atol=0.01) and np.isclose(orig_quat[1], 0.7071, atol=0.01):
            # This is the problematic rotation - extract just the Z component if any
            # For the common case, simply use identity quaternion
            box.orientation.w = 1.0
            box.orientation.x = 0.0
            box.orientation.y = 0.0
            box.orientation.z = 0.0
            self.get_logger().info(f"Corrected problematic X-axis rotation for box {box.id}")
        # If it's a different rotation, keep it as is
        
        self.get_logger().info(f"Plotting box {box.id}: pos=({x},{y},{z}), dims=({length},{width},{height})")
        
        # Define the vertices of the box (relative to corner)
        verts = [
            [x, y, z],                         # bottom front left
            [x + length, y, z],                # bottom front right
            [x + length, y + width, z],        # bottom back right
            [x, y + width, z],                 # bottom back left
            [x, y, z + height],                # top front left
            [x + length, y, z + height],       # top front right
            [x + length, y + width, z + height], # top back right
            [x, y + width, z + height],        # top back left
        ]
        
        # Print initial vertices
        self.get_logger().info(f"Initial vertices for box {box.id}:")
        for i, v in enumerate(verts):
            self.get_logger().info(f"  Vertex {i}: ({v[0]}, {v[1]}, {v[2]})")
        
        # Only apply quaternion rotation if it's not the identity quaternion
        quat = [box.orientation.w, box.orientation.x, box.orientation.y, box.orientation.z]
        if not np.isclose(quat[0], 1.0) or not np.allclose(quat[1:], [0, 0, 0]):
            # We have a real rotation to apply
            self.get_logger().info(f"Applying rotation: w={quat[0]}, x={quat[1]}, y={quat[2]}, z={quat[3]}")
            
            # Convert quaternion to rotation matrix
            rotation_matrix = quat2mat(quat)
            self.get_logger().info(f"Rotation matrix:\n{rotation_matrix}")
            
            # Box center for rotation
            center = np.array([
                x + length/2,
                y + width/2,
                z + height/2
            ])
            self.get_logger().info(f"Rotation center: ({center[0]}, {center[1]}, {center[2]})")
            
            # Apply rotation around box center
            rotated_verts = []
            for i, v in enumerate(verts):
                v_array = np.array(v)
                # Vector from center to vertex
                v_centered = v_array - center
                # Apply rotation
                v_rotated = np.dot(rotation_matrix, v_centered)
                # Translate back
                v_final = v_rotated + center
                rotated_verts.append(v_final.tolist())
                self.get_logger().info(f"  Vertex {i}: Original ({v[0]}, {v[1]}, {v[2]}) -> Rotated ({v_final[0]}, {v_final[1]}, {v_final[2]})")
            
            verts = rotated_verts
        
        # Print final vertices
        self.get_logger().info(f"Final vertices for box {box.id}:")
        for i, v in enumerate(verts):
            self.get_logger().info(f"  Vertex {i}: ({v[0]}, {v[1]}, {v[2]})")
        
        # Get consistent color for this box ID
        box_color = self.get_box_color(box.id)
        
        # Define faces using the standard order
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],  # bottom
            [verts[4], verts[5], verts[6], verts[7]],  # top
            [verts[0], verts[1], verts[5], verts[4]],  # front
            [verts[1], verts[2], verts[6], verts[5]],  # right
            [verts[2], verts[3], verts[7], verts[6]],  # back
            [verts[3], verts[0], verts[4], verts[7]],  # left
        ]
        
        # Log face information
        for i, face in enumerate(faces):
            self.get_logger().info(f"Face {i}: {face}")
        
        # Plot the box using Poly3DCollection
        collection = Poly3DCollection(faces, facecolors=box_color, edgecolors='k', alpha=0.7)
        self.ax.add_collection3d(collection)
        
        # Add box ID label near the top center of the box
        label_x = x + length/2
        label_y = y + width/2
        label_z = z + height + 2  # Slightly above box
        
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
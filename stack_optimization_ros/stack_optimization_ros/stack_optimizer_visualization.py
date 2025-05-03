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
        """Callback for box state messages with added debugging"""
        with self.data_lock:
            self.boxes = msg.boxes
            self.update_needed = True
            
            # Debug logging
            self.get_logger().info(f"Received box state with {len(self.boxes)} boxes")
            
            # Log details of each box
            for box in self.boxes:
                self.get_logger().info(f"Box {box.id}: Position ({box.x}, {box.y}, {box.z}), " + 
                                    f"Dimensions: {box.width}x{box.length}x{box.height}")
                self.get_logger().info(f"Orientation: w={box.orientation.w}, x={box.orientation.x}, " +
                                    f"y={box.orientation.y}, z={box.orientation.z}")
            
            # Calculate current max dimensions with extra buffer
            if self.boxes:
                # Only expand limits, never contract
                current_x_max = max([box.x + box.width for box in self.boxes]) * 1.2
                current_y_max = max([box.y + box.length for box in self.boxes]) * 1.2
                current_z_max = max([box.z + box.height for box in self.boxes]) * 1.2
                
                # Update max values only if they increase significantly (avoid small changes)
                if current_x_max > self.x_max * 1.1:
                    self.x_max = current_x_max
                if current_y_max > self.y_max * 1.1:
                    self.y_max = current_y_max
                if current_z_max > self.z_max * 1.1:
                    self.z_max = current_z_max

    def plot_box(self, box):
        """Visualize a single box in 3D with proper quaternion orientation application"""
        # Box dimensions
        width = box.width
        length = box.length
        height = box.height
        
        # Box position (bottom corner)
        x = box.x
        y = box.y
        z = box.z
        
        # Log coordinates for debugging
        self.get_logger().info(f"Plotting box {box.id}: pos=({x},{y},{z}), dims=({width},{length},{height})")
        
        # Create quaternion from the orientation - convert to proper format
        # The server uses [x,y,z,w] format internally but ROS messages use [w,x,y,z]
        quat = [box.orientation.w, box.orientation.x, box.orientation.y, box.orientation.z]
        
        # Define the vertices of the box before rotation (relative to box corner)
        verts_local = [
            [0, 0, 0],                # bottom front left
            [length, 0, 0],           # bottom front right - length along X-axis
            [length, width, 0],       # bottom back right
            [0, width, 0],            # bottom back left - width along Y-axis
            [0, 0, height],           # top front left
            [length, 0, height],      # top front right
            [length, width, height],  # top back right
            [0, width, height],       # top back left
]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quat2mat(quat)
        
        # Apply rotation to each vertex and add the box position
        verts = []
        for v in verts_local:
            # Apply rotation
            v_rotated = np.dot(rotation_matrix, v)
            # Add box position
            v_positioned = [
                x + v_rotated[0],
                y + v_rotated[1],
                z + v_rotated[2]
            ]
            verts.append(v_positioned)
        
        # Get consistent color for this box ID
        box_color = self.get_box_color(box.id)
        
        # Define faces using the server's face ordering
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],  # bottom
            [verts[4], verts[5], verts[6], verts[7]],  # top
            [verts[0], verts[1], verts[5], verts[4]],  # front
            [verts[1], verts[2], verts[6], verts[5]],  # right
            [verts[2], verts[3], verts[7], verts[6]],  # back
            [verts[3], verts[0], verts[4], verts[7]],  # left
        ]
        
        # Plot the box using Poly3DCollection
        collection = Poly3DCollection(faces, facecolors=box_color, edgecolors='k', alpha=0.7)
        self.ax.add_collection3d(collection)
        
        # Add box ID label near the top center of the box
        # Apply rotation to find the actual center top
        center_top_local = [width/2, length/2, height]
        center_top_rotated = np.dot(rotation_matrix, center_top_local)
        label_x = x + center_top_rotated[0]
        label_y = y + center_top_rotated[1]
        label_z = z + center_top_rotated[2]
        
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
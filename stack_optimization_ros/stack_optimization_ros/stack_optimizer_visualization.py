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
from matplotlib.colors import TABLEAU_COLORS
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class BoxStackVisualizer(Node):
    def __init__(self):
        super().__init__('box_stack_visualizer')
        
        # Initialize state
        self.boxes = []
        self.data_lock = threading.Lock()
        self.update_needed = False
        self.first_draw = True
        
        # Visualization parameters
        self.margin = 10  # Match server margin
        
        # Initialize scaling boundaries
        self.x_max = 80.0
        self.y_max = 60.0
        self.z_max = 100.0
        
        # Create color map for consistent box colors by ID
        base_colors = list(TABLEAU_COLORS.values())
        self.colors = base_colors
        
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
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)
        
        self.get_logger().info('Box stack visualizer started')

    def box_state_callback(self, msg):
        with self.data_lock:
            self.boxes = msg.boxes
            self.update_needed = True

    def plot_box(self, box):
        # Treat box.x, box.y, box.z as bottom-corner origin
        x = box.x
        y = box.y
        z = box.z
        width = box.width
        length = box.length
        height = box.height

        # Build vertices: X=width, Y=length, Z=height
        verts = [
            [x,           y,          z],
            [x+width,     y,          z],
            [x+width,     y+length,   z],
            [x,           y+length,   z],
            [x,           y,          z+height],
            [x+width,     y,          z+height],
            [x+width,     y+length,   z+height],
            [x,           y+length,   z+height],
        ]

        # Faces (same order as server)
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],
            [verts[4], verts[5], verts[6], verts[7]],
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[1], verts[2], verts[6], verts[5]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[3], verts[0], verts[4], verts[7]],
        ]

        # Plot box
        color = self.colors[box.id % len(self.colors)]
        poly = Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=0.7)
        self.ax.add_collection3d(poly)

        # Label at top-center
        cx = x + width/2
        cy = y + length/2
        cz = z + height + 2
        self.ax.text(cx, cy, cz, f"ID: {box.id}", color='k', backgroundcolor=color)

        # Draw margin outline
        m = self.margin
        mx0 = x - m
        my0 = y - m
        mz0 = z
        dx_m = width + 2*m
        dy_m = length + 2*m
        dz_m = height
        # bottom rectangle
        self.ax.plot([mx0, mx0+dx_m], [my0, my0], [mz0, mz0], 'k--', alpha=0.3)
        self.ax.plot([mx0, mx0], [my0, my0+dy_m], [mz0, mz0], 'k--', alpha=0.3)
        self.ax.plot([mx0+dx_m, mx0+dx_m], [my0, my0+dy_m], [mz0, mz0], 'k--', alpha=0.3)
        self.ax.plot([mx0, mx0+dx_m], [my0+dy_m, my0+dy_m], [mz0, mz0], 'k--', alpha=0.3)
        # top rectangle
        mz1 = mz0 + dz_m
        self.ax.plot([mx0, mx0+dx_m], [my0, my0], [mz1, mz1], 'k--', alpha=0.3)
        self.ax.plot([mx0, mx0], [my0, my0+dy_m], [mz1, mz1], 'k--', alpha=0.3)
        self.ax.plot([mx0+dx_m, mx0+dx_m], [my0, my0+dy_m], [mz1, mz1], 'k--', alpha=0.3)
        self.ax.plot([mx0, mx0+dx_m], [my0+dy_m, my0+dy_m], [mz1, mz1], 'k--', alpha=0.3)

    def setup_plot(self):
        self.ax.set_xlabel('X (Width)')
        self.ax.set_ylabel('Y (Length)')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Box Stack Visualization')
        self.ax.set_xlim(0, self.x_max)
        self.ax.set_ylim(0, self.x_max)
        self.ax.set_zlim(0, self.z_max)
        if self.first_draw:
            self.ax.view_init(elev=30, azim=45)
            self.first_draw = False

    def update_plot(self, frame):
        with self.data_lock:
            if not self.update_needed:
                return
            self.ax.clear()
            self.setup_plot()
            self.ax.bar3d(0, 0, 0,
                         self.x_max, self.y_max, 0.1,
                         color='yellow', alpha=0.1, shade=True, zorder=0)
            for box in self.boxes:
                self.plot_box(box)
            self.update_needed = False

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.azim = self.ax.azim
            self.elev = self.ax.elev


def main(args=None):
    rclpy.init(args=args)
    vis = BoxStackVisualizer()
    try:
        threading.Thread(target=rclpy.spin, args=(vis,), daemon=True).start()
        plt.show()
    finally:
        vis.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import blenderproc as bproc
import numpy as np
import json
import cv2

# Initialize BlenderProc
bproc.init()

# Load the .glb model
objs = bproc.loader.load_obj("CAD/cardboard_box_v2.obj")

# Enable physics (optional for realistic positioning)
for obj in objs:
    obj.enable_rigidbody(active=True)
    
# Set up the camera resolution and intrinsics
fx = 600  
fy = 600  
cx = 320  
cy = 240  
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

bproc.camera.set_resolution(640, 480)
bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)

# Camera extrinsic parameters
cam_pose = np.eye(4)
cam_pose[:3, 3] = np.array([0, -1, 10])  # Move camera back slightly to see objects
bproc.camera.add_camera_pose(cam_pose)

print("Camera Pose:\n", cam_pose)  # Debugging


# Use a Sun Light instead of Point Light
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0, -3, 3])  # Move light back
light.set_energy(0.5)  # Increase brightness

# Enable output for RGB, Depth, and Segmentation
bproc.renderer.enable_depth_output(True)
bproc.renderer.enable_normals_output()

# Render images
data = bproc.renderer.render()

# Save debug image to check output

cv2.imwrite("debug_image.png", cv2.cvtColor(data["colors"][0], cv2.COLOR_BGR2RGB))  

# Save the results
bproc.writer.write_hdf5("BlenderProc/output/", data)

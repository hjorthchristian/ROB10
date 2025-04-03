import socket
import time

# Robot configuration
ROBOT_IP = "192.168.1.1"  
PORT = 9001  # Secondary client interface for URScript commands

def send_command(s, command):
    # Add newline to execute command
    cmd = command + "\n"
    s.send(cmd.encode())
    time.sleep(0.5)  # Brief pause for command processing

# Create socket connection
print(f'Connecting to robot at {ROBOT_IP}:{PORT}...')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ROBOT_IP, PORT))

# Example commands for VG10 gripper
try:
    # Grip with both channels at 60% vacuum, 5s timeout
    print("Gripping with both channels...")
    send_command(s, "vg10_grip(2, 60, 5, True, vg_index_get())")
    time.sleep(3)  # Wait for grip to complete
    
    # Release both channels
    print("Releasing with both channels...")
    send_command(s, "vg10_release(2, 5, True, vg_index_get())")
    
finally:
    # Always close the connection
    s.close()

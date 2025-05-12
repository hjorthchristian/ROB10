import rclpy
from rclpy.node import Node
from pymodbus.client import ModbusTcpClient
from std_msgs.msg import Int32, Bool
from std_srvs.srv import Trigger
# Import our custom services
from vg_control_interfaces.srv import VacuumSet, VacuumRelease
import time

class VGGripperNode(Node):
    def __init__(self):
        super().__init__('vg10_gripper_node')
        
        # Parameters
        self.declare_parameter('ip', '192.168.1.1')
        self.declare_parameter('port', 502)
        self.declare_parameter('changer_addr', 65)
        
        # Get parameters
        self.ip = self.get_parameter('ip').value
        self.port = self.get_parameter('port').value
        self.changer_addr = self.get_parameter('changer_addr').value
        
        # Create Modbus client
        self.client = None
        self.connect_modbus()
        
        # Services
        # New service for adjustable grip
        self.srv_grip_adjust = self.create_service(VacuumSet, 'grip_adjust', self.grip_adjust_callback)
        # New service for vacuum release
        self.srv_release = self.create_service(VacuumRelease, 'release_vacuum', self.release_callback)

        # Subscribers
        self.sub_channel_a = self.create_subscription(
            Int32, 
            'set_vacuum_a', 
            lambda msg: self.set_vacuum_callback(msg, channel=0),
            10)
        self.sub_channel_b = self.create_subscription(
            Int32, 
            'set_vacuum_b', 
            lambda msg: self.set_vacuum_callback(msg, channel=1),
            10)
            
        # Publishers for vacuum levels
        self.pub_vacuum_a = self.create_publisher(Int32, 'vacuum_level_a', 20)
        self.pub_vacuum_b = self.create_publisher(Int32, 'vacuum_level_b', 20)
        
        # Timer for reading vacuum levels periodically
        self.create_timer(0.05, self.read_vacuum_levels)
        
        self.get_logger().info(f'VG Gripper Node started, connected to {self.ip}:{self.port}')

    def connect_modbus(self):
        """Connect to the Modbus client"""
        try:
            if self.client is not None:
                self.client.close()
                
            self.client = ModbusTcpClient(
                self.ip,
                port=self.port,
                timeout=1
            )
            success = self.client.connect()
            if success:
                self.get_logger().info(f"Connected to robot at {self.ip}:{self.port}")
            else:
                self.get_logger().error(f"Failed to connect to {self.ip}:{self.port}")
                
            return success
        except Exception as e:
            self.get_logger().error(f"Connection error: {str(e)}")
            return False

    def grip_adjust_callback(self, request, response):
        """Grip with both channels at specified vacuum levels"""
        try:
            # Ensure vacuum levels are within valid range
            vacuum_a = max(0, min(255, request.channel_a))
            vacuum_b = max(0, min(255, request.channel_b))
            
            # Mode: grip (0x0100) + Vacuum: specified levels
            command = [0x0100 + vacuum_a, 0x0100 + vacuum_b]
            self.client.write_registers(address=0, values=command, slave=self.changer_addr)
            
            self.get_logger().info(f"Gripping with channels A:{vacuum_a}, B:{vacuum_b}")
            response.success = True
            response.message = f"Gripping with vacuum levels A:{vacuum_a}, B:{vacuum_b}"
        except Exception as e:
            self.get_logger().error(f"Grip adjust error: {str(e)}")
            response.success = False
            response.message = str(e)
        return response

    def release_callback(self, request, response):
        """Release vacuum on both channels"""
        try:
            if request.release_vacuum == 1:
                # Mode: release (0x0000) + Vacuum: 0
                command = [0x0000, 0x0000]
                self.client.write_registers(address=0, values=command, slave=self.changer_addr)
                
                self.get_logger().info("Released vacuum on both channels")
                response.success = True
                response.message = "Vacuum released successfully"
            else:
                response.success = False
                response.message = "Invalid release_vacuum value. Use 1 to release."
        except Exception as e:
            self.get_logger().error(f"Release vacuum error: {str(e)}")
            response.success = False
            response.message = str(e)
        return response

    def set_vacuum_callback(self, msg, channel):
        """Set vacuum level for a specific channel"""
        try:
            # Ensure vacuum level is within valid range
            vacuum_level = max(0, min(255, msg.data))
            
            # Read current values
            response = self.client.read_holding_registers(address=0, count=2, slave=self.changer_addr)
            if response.isError():
                self.get_logger().error(f"Error reading current vacuum settings: {response}")
                return
                
            current_values = response.registers
            
            # Update only the requested channel, preserve mode bits
            if channel == 0:  # Channel A
                command = [0x0100 + vacuum_level, current_values[1]]
            else:  # Channel B
                command = [current_values[0], 0x0100 + vacuum_level]
                
            self.client.write_registers(address=0, values=command, slave=self.changer_addr)
            
            self.get_logger().info(f"Set vacuum channel {channel} to {vacuum_level}")
        except Exception as e:
            self.get_logger().error(f"Set vacuum error: {str(e)}")

    def read_vacuum_levels(self):
        """Read and publish current vacuum levels"""
        try:
            response = self.client.read_holding_registers(address=258, count=2, slave=self.changer_addr)
            if response.isError():
                self.get_logger().error(f"Error reading vacuum levels: {response}")
                return
                
            registers = response.registers
            
            # Publish vacuum levels
            msg_a = Int32()
            msg_a.data = registers[0]
            self.pub_vacuum_a.publish(msg_a)
            
            msg_b = Int32()
            msg_b.data = registers[1]
            self.pub_vacuum_b.publish(msg_b)
            
            self.get_logger().debug(f"Vacuum levels - A: {registers[0]}, B: {registers[1]}")
        except Exception as e:
            self.get_logger().error(f"Read vacuum levels error: {str(e)}")

    def destroy_node(self):
        """Clean up resources when node is shutting down"""
        if self.client and self.client.is_socket_open():
            # Make sure to release before shutting down
            try:
                command = [0x0000, 0x0000]
                self.client.write_registers(address=0, values=command, slave=self.changer_addr)
                self.get_logger().info("Released gripper on shutdown")
            except Exception as e:
                self.get_logger().error(f"Error releasing on shutdown: {str(e)}")
            finally:
                self.client.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = VGGripperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import time
from pymodbus.client import ModbusTcpClient

def control_vg10(ip="192.168.1.1", port=502, changer_addr=65):
    # Create and connect Modbus client
    client = ModbusTcpClient(
        ip,
        port=port,
        timeout=1
    )
    client.connect()
    
    print("Connected to robot at {}:{}".format(ip, port))
    
    try:
        # Grip with both channels
        print("Gripping with both channels...")
        # Mode: grip (0x0100) + Vacuum: full (255)
        command = [0x0100 + 120, 0x0100 + 120]
        client.write_registers(address=0, values=command, slave=changer_addr)
        
        # Wait a moment
        time.sleep(2)
        
        # Read current vacuum levels
        response = client.read_holding_registers(address=258, count=2, slave=changer_addr).registers
        print(f"Current vacuum levels - Channel A: {response[0]}, Channel B: {response[1]}")
        
        # Wait a moment
        time.sleep(1)
        
        # Release with both channels
        print("Releasing with both channels...")
        # Mode: release (0x0000) + Vacuum: none (0)
        command = [0x0000 + 0, 0x0000 + 0]
        client.write_registers(address=0, values=command, slave=changer_addr)
        
    finally:
        # Close connection
        client.close()

if __name__ == "__main__":
    control_vg10()
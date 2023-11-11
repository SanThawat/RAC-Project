import rtde_control
import time
import rtde_io
import math
import keyboard

# Initialize RTDEControlInterface and RTDEIOInterface
rtde_c = rtde_control.RTDEControlInterface("192.168.20.35")
rtdeio = rtde_io.RTDEIOInterface("192.168.20.35")

# Define initial position and step size
speed = 1
acceleration = 1
height = 396
x, y, z = 260, 105, height
step = 10  # Initial step size

def convert(input_list):
    if len(input_list) != 6:
        return "Input list must contain exactly 6 elements."

    part1 = [x / 1000 for x in input_list[:3]]
    part2 = [x * (math.pi / 180) for x in input_list[3:]]
    output_list = part1 + part2
    return output_list

def print_coordinates(x, y, z):
    print(f"Current coordinates: X={x}, Y={y}, Z={z}")

# Function to update the robot's position and print coordinates
def update_position(dx, dy, dz):
    global x, y, z
    x += dx
    y += dy
    z += dz
    pos = convert([x, y, z, 90, 160, 0])
    rtde_c.moveJ_IK(pos, speed, acceleration)
    print_coordinates(x, y, z)


# Function to update the robot's position with a smaller step (1 unit)
def update_position_small_step(dx, dy, dz):
    global x, y, z, step
    x += dx  # Multiply by the step size
    y += dy
    z += dz
    pos = convert([x, y, z, 90, 160, 0])
    rtde_c.moveJ_IK(pos, speed, acceleration)
    print_coordinates(x, y, z)

# Register hotkeys to control the robot
keyboard.add_hotkey('w', update_position, args=(step, 0, 0))
keyboard.add_hotkey('s', update_position, args=(-step, 0, 0))
keyboard.add_hotkey('a', update_position, args=(0, -step, 0))
keyboard.add_hotkey('d', update_position, args=(0, step, 0))
keyboard.add_hotkey('t', update_position, args=(0, 0, step))
keyboard.add_hotkey('y', update_position, args=(0, 0, -step))

# Register hotkeys to control the robot with a smaller step (Shift + W/A/S/D/T/Y)
keyboard.add_hotkey('shift + w', update_position_small_step, args=(1, 0, 0))
keyboard.add_hotkey('shift + s', update_position_small_step, args=(-1, 0, 0))
keyboard.add_hotkey('shift + a', update_position_small_step, args=(0, -1, 0))
keyboard.add_hotkey('shift + d', update_position_small_step, args=(0, 1, 0))
keyboard.add_hotkey('shift + t', update_position_small_step, args=(0, 0, 1))
keyboard.add_hotkey('shift + y', update_position_small_step, args=(0, 0, -1))

print("Use 'W', 'S', 'A', 'D', 'T', and 'Y' keys to control the robot.")
print("Press 'Ctrl+C' to exit.")
print("Hold 'Shift' while pressing 'W', 'A', 'S', 'D', 'T', or 'Y' for smaller step size.")

while True:
    try:
        keyboard.read_event()
    except KeyboardInterrupt:
        break


# Release resources
keyboard.unhook_all()
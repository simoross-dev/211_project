
#!/usr/bin/env python3

# --- Numpy Compatibility Patch ---
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'maximum_sctype'):
    np.maximum_sctype = lambda t: t
# ---------------------------------

import math
import rclpy
from rclpy.node import Node
import tf_transformations
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data

class BumpAndGoController(Node):
    def __init__(self):
        super().__init__('bump_and_go_controller')

        # Parameters
        self.declare_parameter('max_linear_vel', 0.2)           # TB3 Burger max is 0.22 m/s
        self.declare_parameter('max_angular_vel', 1.0)          
        self.declare_parameter('control_loop_frequency', 10.0)  
        self.declare_parameter('obstacle_threshold', 0.3)       # Set to your 0.3m target

        self.max_lin_vel = self.get_parameter('max_linear_vel').value
        self.max_ang_vel = self.get_parameter('max_angular_vel').value
        self.freq = self.get_parameter('control_loop_frequency').value
        self.threshold = self.get_parameter('obstacle_threshold').value

        # Subscribers and Publishers
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # State Variables
        self.state = 'GO_FORWARD'
        self.turn_direction = 1  
        self.start_yaw = 0.0
        self.current_yaw = 0.0

        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')

        self.timer = self.create_timer(1.0 / self.freq, self.control_loop)
        self.get_logger().info("Hardware-Robust Bump&Go Controller Started.")

    def odom_callback(self, msg):
        """Converts quaternion to Euler yaw."""
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion(quat)

    def scan_callback(self, msg):
        """Dynamically calculates LiDAR indices for the real robot hardware."""
        front_ranges, left_ranges, right_ranges = [], [], []

        for i, r in enumerate(msg.ranges):
            # Ignore invalid readings
            if math.isnan(r) or r <= msg.range_min:
                continue
            
            if math.isinf(r) or r >= msg.range_max:
                r = msg.range_max
            
            # Dynamically calculate angle using sensor metadata
            angle = msg.angle_min + i * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle)) # Normalize to [-pi, pi]

            # Group ranges into ±15 degree directional cones
            if abs(angle) <= math.radians(15):
                front_ranges.append(r)
            elif abs(angle - math.radians(90)) <= math.radians(15):
                left_ranges.append(r)
            elif abs(angle + math.radians(90)) <= math.radians(15):
                right_ranges.append(r)

        # Use min() instead of average to detect thin obstacles instantly
        self.front_dist = min(front_ranges) if front_ranges else msg.range_max
        self.left_dist = min(left_ranges) if left_ranges else msg.range_max
        self.right_dist = min(right_ranges) if right_ranges else msg.range_max

    def control_loop(self):
        twist = Twist()

        if self.state == 'GO_FORWARD':
            if self.front_dist <= self.threshold:
                self.state = 'TURN'
                self.start_yaw = self.current_yaw
                
                # Turn toward the side with more open space
                self.turn_direction = 1 if self.left_dist >= self.right_dist else -1
                dir_str = "LEFT" if self.turn_direction == 1 else "RIGHT"
                self.get_logger().info(f"Obstacle at {self.front_dist:.2f}m! Turning {dir_str}.")

                twist.linear.x = 0.0
                twist.angular.z = self.turn_direction * self.max_ang_vel
            else:
                twist.linear.x = self.max_lin_vel
                twist.angular.z = 0.0

        elif self.state == 'TURN':
            # Stop turning when the front is clear beyond the threshold + 0.1m hysteresis margin
            if self.front_dist > self.threshold + 0.1:
                # Calculate normalized rotation amount for logging
                rotated_angle = math.atan2(math.sin(self.current_yaw - self.start_yaw), 
                                           math.cos(self.current_yaw - self.start_yaw))
                self.get_logger().info(f"Path free! Rotated {math.degrees(abs(rotated_angle)):.1f} deg.")
                
                self.state = 'GO_FORWARD'
                twist.linear.x = self.max_lin_vel
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                twist.angular.z = self.turn_direction * self.max_ang_vel

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = BumpAndGoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
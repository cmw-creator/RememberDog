#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math

class NavClient(Node):
    def __init__(self):
        super().__init__('nav_client')
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def send_goal(self, x, y, yaw=0.0):
        """发送导航目标点 (x, y, yaw[弧度])"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._client.wait_for_server()
        return self._client.send_goal_async(goal_msg)

def go_to(x, y, yaw=0.0):
    """外部接口：阻塞式发送导航目标"""
    rclpy.init()
    node = NavClient()

    future = node.send_goal(x, y, yaw)
    rclpy.spin_until_future_complete(node, future)

    node.destroy_node()
    rclpy.shutdown()
    print(f"导航目标已发送: ({x:.2f}, {y:.2f}, {yaw:.2f})")

import os

feedback = os.system ("ros2 topic pub /move_base_simple/goal geometry_msgs/PoseStamped \"{header: {stamp: {sec: 1}, frame_id: 'map'}, pose: {position: {x: 0., y: -0.1, z: 0.5}, orientation: {w: 0.5}}}\"")


import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yousefabdelhady/SESASR_labs/lab_4/ros2_ws/install/lab04_pkg'

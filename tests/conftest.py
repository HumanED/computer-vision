"""
pytest configuration for my_script tests.

Adds the project root (mock_camera.py) and src/ (person_follower, go_home)
to sys.path so tests can import them directly.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))        # my_script/ — for mock_camera
sys.path.insert(0, os.path.join(_HERE, "..", "src")) # my_script/src/ — for person_follower, go_home

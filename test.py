import sys
import os

print(f"1. Python Interpreter: {sys.executable}")

try:
    import lerobot
    print(f"2. SUCCESS! LeRobot found at: {os.path.dirname(lerobot.__file__)}")
except ImportError as e:
    print(f"2. FAILURE: {e}")
    print("   (Python cannot find the library files, even though pip lists them.)")

except AttributeError:
    print("2. PARTIAL FAILURE: Found 'lerobot' folder, but it seems empty or wrong.")
    print("   Check if you have a folder named 'lerobot' in your project directory!")
#!/usr/bin/env python3
import subprocess
from functools import wraps

def log_tool_io(func):
    """Decorator to log tool input and output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Format input arguments
        input_str = ""
        if args:
            input_str += f"args={args}"
        if kwargs:
            if input_str:
                input_str += ", "
            input_str += f"kwargs={kwargs}"
        
        print(f"🔧 TOOL INPUT [{func.__name__}]: {input_str}")
        
        try:
            result = func(*args, **kwargs)
            print(f"✅ TOOL OUTPUT [{func.__name__}]: {result}")
            return result
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ TOOL ERROR [{func.__name__}]: {error_msg}")
            raise
    return wrapper


@log_tool_io
def execute_terminal_command(command: str) -> str:
    if not command or not command.strip():
        return "Error: Empty command provided"

    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        output = result.stdout.strip()
        return output if output else f"Command '{command}' executed successfully (no output)"
    error = result.stderr.strip()
    return f"Command failed with exit code {result.returncode}: {error}" if error else f"Command failed with exit code {result.returncode}"



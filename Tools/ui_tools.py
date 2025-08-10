#!/usr/bin/env python3
import os
import re
import subprocess
import tempfile
from typing import Optional
from functools import wraps

from ocrmac import ocrmac
import pyautogui

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

# Cache last OCR results to optionally support future features
_last_ocr_results = []


@log_tool_io
def read_screen() -> str:
    """Capture the screen and return recognized text via Apple OCR."""
    global _last_ocr_results

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        screenshot_path = temp_file.name

    subprocess.run(["screencapture", "-x", screenshot_path], check=True)
    try:
        annotations = ocrmac.OCR(screenshot_path).recognize()
    finally:
        try:
            os.unlink(screenshot_path)
        except Exception:
            pass

    if not annotations:
        _last_ocr_results = []
        text = "Screen captured but no text detected"
    else:
        confidence_threshold = 0.3
        _last_ocr_results = [
            {"text": a[0], "confidence": a[1], "bbox": a[2]}
            for a in annotations
            if a[1] > confidence_threshold
        ]
        if not _last_ocr_results:
            text = "Text detected but confidence too low (all below 30%)"
        else:
            text_lines = [r["text"] for r in _last_ocr_results]
            text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
            text = f"Screen text: {text}."

    # Update current screen content in the main module if available
    try:
        import sys
        if "Jeeves" in sys.modules:
            sys.modules["Jeeves"].current_screen_content = text
    except Exception:
        pass

    return text


@log_tool_io
def type_text(text: str) -> str:
    """Type text using the keyboard."""
    if not text or not text.strip():
        return "Error: No text provided to type"
    pyautogui.write(text)
    return f"Typed: '{text}'"



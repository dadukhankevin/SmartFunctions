#!/usr/bin/env python3

import subprocess
import tempfile
import os
import re
import time
import sys
import threading
import queue
import multiprocessing
import json
from ocrmac import ocrmac
import pyautogui
import pyperclip
from dotenv import load_dotenv
from Speech import speak
import agent_comm
from AgentUse.agentuse import AgentUse
from memory_system import get_memory_system
from openai import OpenAI
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

# Global storage for last OCR results
last_ocr_results = []

# Global storage for active agents
active_agents = {}
agent_communication_queues = {}
current_agent_question = None  # Store the most recent agent question
agent_question_queue = []  # Queue for multiple agent questions
jeeves_notification_queue = []  # Queue for notifying Jeeves of agent events

class MacOSUI:
    """Minimal macOS interactions: screen reading + typing only"""

    def __init__(self):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    # === SCREEN READING ===
    @log_tool_io
    def read_screen(self):
        global last_ocr_results
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                screenshot_path = temp_file.name
            
            result = subprocess.run(['screencapture', '-x', screenshot_path], capture_output=True, text=True)
            if result.returncode != 0:
                return f"Failed to capture screen: {result.stderr}"
            
            if not os.path.exists(screenshot_path):
                return "Screenshot file was not created"
                
            file_size = os.path.getsize(screenshot_path)
            print(f"🖼️ Screenshot saved to: {screenshot_path} (size: {file_size} bytes)")
            
            annotations = ocrmac.OCR(screenshot_path).recognize()
            
            print(f"📝 OCR annotations: {annotations}")
            os.unlink(screenshot_path)
            
            if not annotations:
                last_ocr_results = []
                return "Screen captured but no text detected"
            
            confidence_threshold = 0.3
            last_ocr_results = [
                {
                    'text': annotation[0], 
                    'confidence': annotation[1], 
                    'bbox': annotation[2]
                } 
                for annotation in annotations if annotation[1] > confidence_threshold
            ]
            
            if not last_ocr_results:
                return "Text detected but confidence too low (all below 30%)"
            
            text_lines = [result['text'] for result in last_ocr_results]
            full_text = re.sub(r'\s+', ' ', ' '.join(text_lines)).strip()
            
            return f"Screen text: {full_text}."
            
        except Exception as e:
            return f"Error reading screen: {str(e)}"
    # === KEYBOARD INPUT ===
    @log_tool_io
    def type_text(self, text):
        if not text.strip():
            return "Error: No text provided to type"
        
        try:
            pyautogui.write(text)
            return f"Typed: '{text}'"
        except Exception as e:
            return f"Error typing text: {str(e)}"

# Create unified UI instance
ui = MacOSUI()

@log_tool_io
def read_screen():
    result = ui.read_screen()
    # Update the global screen content in the main module
    try:
        import sys
        if 'Jeeves' in sys.modules:
            sys.modules['Jeeves'].current_screen_content = result
    except:
        pass  # Fail silently if we can't update the main module
    return result

@log_tool_io
def type_text(text):
    return ui.type_text(text)

@log_tool_io
def read_clipboard():
    try:
        content = pyperclip.paste()
        if not content:
            return "Clipboard is empty"
        
        # Limit clipboard content to avoid overwhelming context
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        
        return f"Clipboard content: {content}"
    except Exception as e:
        return f"Error reading clipboard: {str(e)}"


@log_tool_io
def execute_terminal_command(command):
    if not command.strip():
        return "Error: Empty command provided"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else f"Command '{command}' executed successfully (no output)"
        else:
            error = result.stderr.strip()
            return f"Command failed with exit code {result.returncode}: {error}" if error else f"Command failed with exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"

# Removed blocking wait_for_user_response; arming is handled in tool mapping via detector.arm_next_phrase

@log_tool_io
def speak_text(text):
    """Speak the provided text"""
    speak(text)
    return f"Spoke: '{text}'"

# === WAIT / IDLE ===
@log_tool_io
def wait_for_user():
    """No-op tool to explicitly stop any ongoing follow-up. Returns immediately."""
    return "Waiting"

# === MEMORY CONTROL (Intelligent) ===
@log_tool_io
def intelligent_remember(note_text: str) -> str:
    """Use LLM to reconcile memory: forget outdated similar items and save the new truth.
    Returns a summary including new short id and any forgotten ids.
    """
    note_text = (note_text or "").strip()
    if not note_text:
        return "Error: Nothing to remember."

    memory = get_memory_system()
    # Find top similar memories (compact context)
    candidates = memory.get_relevant_memories(note_text, limit=5, context_messages=0)

    # Build LLM prompt
    payload = {
        "new_statement": note_text,
        "candidates": [
            {
                "id": c.get("id"),
                "short_id": c.get("short_id"),
                # Use the single message content if available
                "content": (c.get("messages", [{}])[0].get("content", "")),
            }
            for c in candidates
        ],
        "instructions": (
            "Given the user's new statement and similar past memories, decide which old memories are now outdated or contradicted. "
            "Output strict JSON with keys: forget_ids (array of ids from candidates to remove), and final_memory (concise corrected single sentence). "
            "Do not include any extra keys. Prefer forgetting directly contradicted items only."
        ),
    }

    # Call same model via OpenRouter
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Fallback: just remember without reconciliation
        sid = memory.remember(note_text)
        return f"Saved memory {sid}. (No API key for intelligent reconciliation)"

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b-thinking-2507",
            messages=[
                {"role": "system", "content": "You are a memory reconciler. Respond with strict JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=8,
            stream=False,
            extra_body={
                "provider": {"order": ["Cerebras"], "allow_fallbacks": True}
            },
        )
        content = (resp.choices[0].message.content or "").strip()
        # Extract JSON
        plan = json.loads(content)
        forget_ids = plan.get("forget_ids", []) if isinstance(plan, dict) else []
        final_memory = plan.get("final_memory", note_text)
    except Exception:
        # On any error, save the provided note
        sid = memory.remember(note_text)
        return f"Saved memory {sid}."

    # Apply forget
    forgotten = []
    for mid in forget_ids:
        result = memory.forget(str(mid))
        if result.startswith("Forgot memory"):
            forgotten.append(result.split()[-1])

    # Save final truth
    new_sid = memory.remember(final_memory)
    if forgotten:
        return f"Updated memory {new_sid}. Forgot: {', '.join(forgotten)}"
    return f"Saved memory {new_sid}."

@log_tool_io
def search_memory_tool(query: str) -> str:
    """Search memories for Jeeves - wrapper for the memory_tools function"""
    from Tools.memory_tools import search_memory
    return search_memory(query)

# === AGENT MANAGEMENT ===

@log_tool_io
def spawn_agent(goal, custom_dir=None):
    """Spawn a new AgentUse instance to handle a specific goal
    Args:
        goal: The goal description for the agent
        custom_dir: Optional custom directory (if None, creates new agent directory)
    """
    global active_agents, agent_communication_queues
    
    goal = goal.strip()
    if not goal:
        return "Error: Need a goal description"
    
    # Generate unique agent ID and determine directory
    agent_id = f"agent_{len(active_agents) + 1}_{int(time.time())}"
    if custom_dir:
        agent_dir = custom_dir
        # Don't create directory if it's custom (assume it exists)
        if not os.path.exists(agent_dir):
            return f"Error: Custom directory '{custom_dir}' does not exist"
    else:
        agent_dir = f"/Users/daniellosey/JeevesHimself/agents/{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)
    
    # Create communication queue
    agent_queue = queue.Queue()
    agent_communication_queues[agent_id] = agent_queue
    
    def ask_user(question):
        """Allow spawned agent to ask user questions through Jeeves"""
        global current_agent_question
        print(f"\n🙋 Agent {agent_id} asks: {question}")
        
        # Save agent question to memory
        try:
            from memory_system import get_memory_system
            memory = get_memory_system()
            memory.save_message("agent", f"Question: {question}", agent_id)
        except:
            pass
        
        question_file = agent_comm.post_question(agent_id, question)
        current_agent_question = {
            'agent_id': agent_id,
            'question': question,
            'timestamp': time.time(),
            'answered': False,
            'response': None,
            'file': str(question_file)
        }
        
        # Notify Jeeves immediately and forcefully
        if 'Jeeves' in sys.modules:
            jeeves_module = sys.modules['Jeeves']
            notification = f"\n🔔 QUESTION: I need your input: '{question}'\nUse <answer_agent>your_answer</answer_agent> to respond."
            if hasattr(jeeves_module, 'passive_callback'):
                print("📢 Triggering Jeeves directly with agent question...")
                try:
                    jeeves_module.passive_callback(notification, [])
                except Exception as e:
                    print(f"Error notifying Jeeves: {e}")
            
            # Also add to notification queue as backup
            global jeeves_notification_queue
            jeeves_notification_queue.append(f"URGENT: I need your input: '{question}'")
        
        response = agent_comm.wait_for_answer(question_file, timeout=300)
        
        # Save the user's response to memory if we got one
        if response:
            try:
                memory.save_message("user", f"Response to agent: {response}")
            except:
                pass
        
        return f"User responds: {response}" if response else "No response received within 5 minutes."
    
    def report_progress(status):
        """Allow agent to report progress back to Jeeves"""
        try:
            global jeeves_notification_queue
            agent_queue.put(f"PROGRESS: {status}")
            print(f"📋 Agent {agent_id} reports: {status}")
            
            # Save progress to memory
            try:
                from memory_system import get_memory_system
                memory = get_memory_system()
                memory.save_message("agent", f"Progress: {status}", agent_id)
            except:
                pass
            
            # Add to Jeeves progress buffer (non-blocking)
            if 'Jeeves' in sys.modules:
                jeeves_module = sys.modules['Jeeves']
                if hasattr(jeeves_module, 'agent_progress_buffer'):
                    progress_msg = f"Agent {agent_id}: {status}"
                    jeeves_module.agent_progress_buffer.append(progress_msg)
                    # Keep buffer size manageable (last 10 updates)
                    if len(jeeves_module.agent_progress_buffer) > 10:
                        jeeves_module.agent_progress_buffer.pop(0)
                    print(f"📝 Added to progress buffer: {progress_msg}")
            
            return "Progress reported to Jeeves"
        except Exception as e:
            error_msg = f"Error reporting progress: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    # Get API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not found in environment"
    
    # HERE'S THE AGENT CREATION YOU WERE LOOKING FOR:
    agent = AgentUse(
        api_key=api_key,
        model="qwen/qwen3-235b-a22b-2507",
        provider_order=["Cerebras"],
        instructions="You are working under Jeeves' supervision. Ask about requirements before starting work."
    )
    
    # Add custom tools
    agent.add_tool("<ask_user>question for the user</ask_user>", ask_user)
    agent.add_tool("<report_progress>status update</report_progress>", report_progress)
    
    # Store agent info
    active_agents[agent_id] = {
        'goal': goal,
        'start_time': time.time(),
        'status': 'STARTING'
    }
    
    def run_agent():
        active_agents[agent_id]['status'] = 'RUNNING'
        # Use clone feature to start with template (only for non-custom directories)
        run_params = {
            'goal': goal, 
            'cli_cmd': "gemini", 
            'time_limit': 1440, 
            'directory': agent_dir
        }
        
        # Only clone from template if using default agent directory
        if not custom_dir:
            run_params['clone_from'] = "/Users/daniellosey/JeevesHimself/agent_template"
        
        agent.run(**run_params)
        active_agents[agent_id]['status'] = 'COMPLETED'
        print(f"✅ Agent {agent_id} completed successfully")
    
    thread = threading.Thread(target=run_agent, daemon=True)
    active_agents[agent_id]['thread'] = thread
    thread.start()
    
    return f"Agent {agent_id} spawned with goal: '{goal}' in directory {agent_dir}"

@log_tool_io
def self_improvement_agent(feature_description):
    """Spawn a specialized agent to improve Jeeves himself in the current directory"""
    feature_description = feature_description.strip()
    if not feature_description:
        return "Error: Need a feature description"
    
    current_dir = "/Users/daniellosey/JeevesHimself"
    goal = f"Improve Jeeves by adding this feature: {feature_description}. Work directly in the Jeeves codebase to implement this enhancement."
    
    return spawn_agent(goal, custom_dir=current_dir)

@log_tool_io
def check_agents():
    """Check status of all active agents"""
    global active_agents, current_agent_question
    
    file_questions = agent_comm.check_for_questions()
    
    if not active_agents and not current_agent_question and not file_questions:
        return "No active agents running and no pending questions."
    
    status_report = "Active Agents Status:\n"
    
    for agent_id, agent_info in list(active_agents.items()):
        elapsed = time.time() - agent_info['start_time']
        status = "RUNNING" if agent_info['thread'].is_alive() else "FINISHED"
        status_report += f"• {agent_id}: {status} ({elapsed:.1f}s) - Goal: {agent_info['goal']}\n"
        
        # Clean up finished agents
        if not agent_info['thread'].is_alive():
            del active_agents[agent_id]
    
    # Show pending questions
    if current_agent_question:
        elapsed = time.time() - current_agent_question['timestamp']
        status_report += f"\n📌 Current Question:\n"
        status_report += f"Agent {current_agent_question['agent_id']} asks: \"{current_agent_question['question']}\" ({elapsed:.1f}s ago)\n"
        status_report += f"Use: <answer_agent>your_answer</answer_agent>\n"
    
    if file_questions:
        status_report += f"\n📁 File-based Questions ({len(file_questions)}):\n"
        for i, (file, data) in enumerate(file_questions[:3]):
            elapsed = time.time() - data['timestamp']
            status_report += f"{i+1}. Agent {data['agent_id']}: \"{data['question']}\" ({elapsed:.1f}s ago)\n"
        status_report += f"Use: <answer_agent>your_answer</answer_agent> to respond.\n"
    
    return status_report.strip()

@log_tool_io
def answer_agent(answer):
    """Answer the current pending agent question"""
    global current_agent_question
    
    answer = answer.strip()
    if not answer:
        return "Error: Please provide an answer."
    
    # Check file-based questions first
    questions = agent_comm.check_for_questions()
    if questions:
        question_file, question_data = questions[0]
        
        if agent_comm.answer_question(question_file, answer):
            agent_id = question_data['agent_id']
            question = question_data['question']
            
            if current_agent_question and current_agent_question.get('file') == str(question_file):
                current_agent_question = None
            
            return f"Answer sent to agent {agent_id}.\nQuestion: \"{question}\"\nAnswer: \"{answer}\""
        else:
            return "Error: Could not send answer to agent."
    
    return "No pending agent question to answer."

@log_tool_io
def stop_agent(agent_id):
    """Stop a specific agent (note: forceful termination not implemented yet)"""
    global active_agents
    
    if agent_id not in active_agents:
        available_agents = list(active_agents.keys())
        return f"Agent {agent_id} not found. Active agents: {available_agents}"
    
    # Note: Thread termination is not clean, but we can mark it for cleanup
    agent_info = active_agents[agent_id]
    if agent_info['thread'].is_alive():
        return f"Agent {agent_id} is still running. It will auto-cleanup when finished. (Forced termination not implemented)"
    else:
        del active_agents[agent_id]
        return f"Agent {agent_id} was already finished and has been cleaned up."

@log_tool_io
def check_agent_notifications():
    """Check for any pending agent question notifications for Jeeves"""
    global jeeves_notification_queue
    
    if not jeeves_notification_queue:
        return None
    
    # Get all urgent question notifications
    notifications = jeeves_notification_queue.copy()
    jeeves_notification_queue.clear()
    
    # Format them nicely
    notification_text = "Urgent Agent Questions:\n"
    for notif in notifications:
        notification_text += f"• {notif}\n"
    
    return notification_text.strip()

@log_tool_io
def complete_task(message):
    return "Task completed"

def process_tool_request(tool_tag, content):
    tools = {
        # Screen reading
        "read_screen": lambda: read_screen(),
        
        # Keyboard input
        "type_text": lambda: type_text(content),
        # Press key removed
        # Gestures removed
        
        # Speech
        "speak": lambda: speak_text(content),
        
        # System
        "terminal": lambda: execute_terminal_command(content),
        "complete": lambda: complete_task(content),
        
        # User interaction
    # Make wait_for_response non-blocking by arming the detector to capture the next phrase
    "wait_for_response": lambda: (lambda t=float(content.strip()) if content.strip() else 12.0: (__import__('sys').modules['Jeeves'].detector.arm_next_phrase(t), "Armed for next reply (non-blocking)"))(),
        "wait_for_user": lambda: wait_for_user(),

        # Memory control (intelligent)
        "remember": lambda: intelligent_remember(content),
        "search_memory": lambda: search_memory_tool(content),
        
        # Agent management
        "spawn_agent": lambda: spawn_agent(content),
        "self_improvement_agent": lambda: self_improvement_agent(content),
        "check_agents": lambda: check_agents(),
        "stop_agent": lambda: stop_agent(content.strip()),
        "answer_agent": lambda: answer_agent(content),
        
        # Legacy alias
        "write_text": lambda: type_text(content)
    }
    
    if tool_tag not in tools:
        available_tools = ', '.join(sorted(tools.keys()))
        return f"Unknown tool '{tool_tag}'. Available tools: {available_tools}"
    
    return tools[tool_tag]()

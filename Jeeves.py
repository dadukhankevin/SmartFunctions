#!/usr/bin/env python3

import os
os.environ.update({
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'TOKENIZERS_PARALLELISM': 'false',
    'OMP_NUM_THREADS': '1'
})

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

import re
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict

from ActiveAssistant import ActiveAssistant
from Speech import speak
from openai import OpenAI
from dotenv import load_dotenv
from Tools.default_tools import process_tool_request
from memory_system import get_memory_system
import pyperclip

load_dotenv()

MEMORY_LIMIT = 50
CONTEXT_MESSAGES = 3
AUTO_CONTINUE_TOOLS = {"read_screen", "type_text", "terminal", "answer_agent", "search_memory"}

@dataclass
class JeevesConfig:
    model: str = "qwen/qwen3-coder"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    timeout: int = 2
    temperature: float = .8
    provider_order: Optional[List[str]] = None
    
    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        self.provider_order = self.provider_order or ["Cerebras"]

class ConversationManager:
    def __init__(self, config: JeevesConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.history = [{"role": "system", "content": ""}]
        self.screen_content = ""
        self.agent_progress: List[str] = []
        self.clipboard_content = ""
        
    def build_system_prompt(self) -> str:
        memories = self._get_relevant_memories()
        
        base_prompt = """

        You are Jeeves, Daniel's voice assistant. Act like Jarvis from Iron Man and Jeeves from Wooster and Jeeves.
        Respond conversationally using these tools:

        <terminal say="nothing">command</terminal>
        <read_screen say="nothing">
        <type_text say="nothing">text</type_text>
        <speak say="nothing">text</speak>
        <wait_for_response say="nothing">
        <wait_for_user say="nothing">
        <remember say="nothing">Daniel Likes something because something.</remember>
        <search_memory say="nothing">when did daniel last go fishing, fish, lakes</search_memory>
        <spawn_agent say="I'll get a specialist on this">goal</spawn_agent>
        <self_improvement_agent say="I'll work on improving myself">feature to add to Jeeves</self_improvement_agent>
        <check_agents say="nothing">
        <stop_agent say="nothing">agent_id</stop_agent>
        <answer_agent say="nothing">answer</answer_agent>
        <complete say="nothing">

        Rules:
        - ONE tool per response
        - Default to say="nothing" to stay silent; speak only when it adds value.
        - Delegate complex projects to agents
        - Simple tasks: use personal tools
        - If something isn't working more than 3 times, then ask Daniel what to do.
        Be conversational and concise. Do not speek IP address outloud, error on the side of not talking.
        And use terminal to control smart home things, all info is in your memory.
        """

        sections = [memories, base_prompt]
        if self.agent_progress:
            sections.append(f"\nAgent Updates:\n" + "\n".join(f"• {p}" for p in self.agent_progress[-5:]))
        if self.screen_content:
            sections.append(f"\nScreen: {self.screen_content}")
        if self.clipboard_content:
            sections.append(f"\nClipboard: {self.clipboard_content}")
            
        return "\n".join(filter(None, sections))
    
    def _get_relevant_memories(self) -> str:
        try:
            memory = get_memory_system()
            recent_content = " ".join(
                re.sub(r'<[^>]+>', '', msg["content"]).strip()
                for msg in self.history[-3:]
                if msg.get("role") in ["user", "assistant"] and msg.get("content")
            )
            
            if recent_content:
                memories = memory.get_relevant_memories(recent_content, MEMORY_LIMIT, CONTEXT_MESSAGES)
                return memory.format_memories_for_prompt(memories) if memories else ""
        except Exception:
            pass
        return ""
    
    def add_user_message(self, content: str):
        try:
            get_memory_system().save_message("user", content)
        except Exception:
            pass
        
        self._update_clipboard_content()
        self.history[0]["content"] = self.build_system_prompt()
        self.history.append({"role": "user", "content": content})
    
    def generate_response(self) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.history,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            max_tokens=-1,
            stream=False,
            extra_body={
                "provider": {
                    "order": self.config.provider_order,
                    "allow_fallbacks": True
                }
            }
        )
        
        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": content})
        
        try:
            get_memory_system().save_message("assistant", content)
        except Exception:
            pass
            
        return content
    
    def generate_response_with_prompt(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})
        return self.generate_response()
    
    def _update_clipboard_content(self):
        try:
            content = pyperclip.paste()
            self.clipboard_content = content[:2000] + "..."
        except Exception:
            self.clipboard_content = ""
    
    def update_screen_content(self, content: str):
        self.screen_content = content
        self.history[0]["content"] = self.build_system_prompt()
    
    def add_agent_progress(self, progress: str):
        self.agent_progress.append(progress)
        if len(self.agent_progress) > 10:
            self.agent_progress.pop(0)
    
    def clear_task_state(self):
        self.screen_content = ""
        self.agent_progress = []
        self.clipboard_content = ""
        self.history[0]["content"] = self.build_system_prompt()
        
        for msg in self.history:
            if msg["role"] in ["user", "assistant"]:
                msg["content"] = re.sub(
                    r'<(\w+)_response>.*?</\1_response>',
                    r'<\1_response>Context cleared after task completion.</\1_response>',
                    msg["content"],
                    flags=re.DOTALL
                )

class ToolProcessor:
    def __init__(self, conversation: ConversationManager, detector):
        self.conversation = conversation
        self.detector = detector
    
    def process_response(self, response: str) -> str:
        tool_match = self._extract_first_tool(response)
        if not tool_match:
            return f"<speak>{response}</speak>"
        
        tag, say_text, content = tool_match
        
        if say_text and say_text.strip().lower() != "nothing":
            self.detector.set_assistant_speaking(True)
            speak(say_text)
            self.detector.set_assistant_speaking(False)
        
        if tag == "complete":
            self.conversation.clear_task_state()
            return ""
            
        if tag == "speak":
            self.detector.set_assistant_speaking(True)
            print(f"🔧 TOOL INPUT [Jeeves->{tag}]: {content}")
            result = process_tool_request(tag, content)
            print(f"✅ TOOL OUTPUT [Jeeves->{tag}]: {result}")
            self.detector.set_assistant_speaking(False)
        else:
            print(f"🔧 TOOL INPUT [Jeeves->{tag}]: {content}")
            result = process_tool_request(tag, content)
            print(f"✅ TOOL OUTPUT [Jeeves->{tag}]: {result}")
        
        if tag == "read_screen":
            self.conversation.update_screen_content(result)
        
        response_tag = f"<{tag}_response>{result}</{tag}_response>"
        
        if tag in AUTO_CONTINUE_TOOLS:
            prompt = f"{response_tag}\n\nContinue with your task or use <complete> if finished."
            return self.process_response(self.conversation.generate_response_with_prompt(prompt))
        elif tag == "check_agents" and ("Current Question:" in result or "File-based Questions" in result):
            prompt = f"{response_tag}\n\nThere's a pending agent question. Answer with <answer_agent> or ask me with <wait_for_response>."
            return self.process_response(self.conversation.generate_response_with_prompt(prompt))
        elif tag == "wait_for_response":
            # Non-blocking: detector is armed; do not auto-continue here
            return response.replace(self._reconstruct_tool_tag(tag, say_text, content), response_tag, 1)
        elif tag == "wait_for_user":
            return ""
        
        return response.replace(self._reconstruct_tool_tag(tag, say_text, content), response_tag, 1)
    
    def _extract_first_tool(self, response: str) -> Optional[tuple]:
        pattern = r'<(\w+)(?:\s+say="([^"]*)")?(?:\s*/>|>(.*?)</\1>|>)'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches[0] if matches else None
    
    def _reconstruct_tool_tag(self, tag: str, say_text: str, content: str) -> str:
        attr = f' say="{say_text}"' if say_text else ''
        return f"<{tag}{attr}>{content}</{tag}>" if content else f"<{tag}{attr}/>"

class JeevesAssistant:
    def __init__(self):
        self.config = JeevesConfig()
        self.conversation = ConversationManager(self.config)
        self.detector = None
        self.processor = None
    
    def passive_callback(self, transcription: str, history: List[Dict]):
        try:
            from Tools.default_tools import current_agent_question, check_agent_notifications
            
            if current_agent_question and not current_agent_question.get('answered'):
                agent_id = current_agent_question['agent_id']
                question = current_agent_question['question']
                print(f"📌 Pending: {agent_id} asks '{question}'")
            
            notifications = check_agent_notifications()
            if notifications:
                transcription = f"{notifications}\n\n{transcription}"
            
            context = "Recent conversation:\n" + "\n".join(
                f"[{entry['timestamp'].strftime('%H:%M:%S')}] {entry['transcription']}"
                for entry in history
            )
            
            full_prompt = f"{context}\nCurrent request: {transcription}"
            self.conversation.add_user_message(full_prompt)
            response = self.conversation.generate_response()
            _ = self.processor.process_response(response)
            
            self.detector.clear_history()
        except Exception as e:
            print(f"❌ Error: {e}")

    def check_agent_events(self):
        while True:
            try:
                from Tools.default_tools import check_agent_notifications, active_agents
                # Only check if there are actually active agents
                if active_agents:
                    notifications = check_agent_notifications()
                    if notifications and self.detector:
                        self.passive_callback(f"AGENT NOTIFICATION: {notifications}", [])
            except Exception:
                pass
            time.sleep(5.0)
    
    def start(self):
        print("JEEVES ACTIVE ASSISTANT")
        print("Listening for requests...")
        
        self.detector = ActiveAssistant(callback=self.passive_callback, verbose=True)
        self.processor = ToolProcessor(self.conversation, self.detector)
        
        # Legacy compatibility for tools that expect global variables
        import sys
        sys.modules['Jeeves'] = type('Module', (), {
            'agent_progress_buffer': self.conversation.agent_progress,
            'current_screen_content': self.conversation.screen_content,
            'detector': self.detector
        })()
        
        threading.Thread(target=self.check_agent_events, daemon=True).start()
        
        speak("Ready.")
        self.detector.start_listening()

if __name__ == "__main__":
    JeevesAssistant().start()
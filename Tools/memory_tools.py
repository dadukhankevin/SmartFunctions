#!/usr/bin/env python3
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from functools import wraps

from memory_system import get_memory_system

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
def remember_intelligent(note_text: str) -> str:
    print("Remembering: ", note_text)
    note_text = (note_text or "").strip()
    if not note_text:
        return "Error: Nothing to remember."

    memory = get_memory_system()
    candidates = memory.get_relevant_memories(note_text, limit=5, context_messages=0)

    payload = {
        "new_statement": note_text,
        "candidates": [
            {
                "id": c.get("id"),
                "short_id": c.get("short_id"),
                "content": (c.get("messages", [{}])[0].get("content", "")),
            }
            for c in candidates
        ],
        "instructions": (
            "Given the user's new statement and similar past memories, decide which old memories are now outdated or contradicted. "
            "Output strict JSON with keys: forget_ids (array of ids from candidates to remove), and final_memory (concise corrected note, may be multiple sentences). "
            "Do not include any extra keys. Prefer forgetting directly contradicted items only."
        ),
    }

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
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
            max_tokens=600,
            timeout=8,
            stream=False,
            extra_body={
                "provider": {"order": ["Cerebras"], "allow_fallbacks": True}
            },
        )
        content = (resp.choices[0].message.content or "").strip()
        plan = json.loads(content)
        forget_ids = plan.get("forget_ids", []) if isinstance(plan, dict) else []
        final_memory = plan.get("final_memory", note_text)
    except Exception:
        sid = memory.remember(note_text)
        return f"Saved memory {sid}."

    forgotten = []
    for mid in forget_ids:
        result = memory.forget(str(mid))
        if result.startswith("Forgot memory"):
            forgotten.append(result.split()[-1])

    new_sid = memory.remember(final_memory)
    if forgotten:
        return f"Updated memory {new_sid}. Forgot: {', '.join(forgotten)}"
    return f"Saved memory {new_sid}."


@log_tool_io
def search_memory(query: str) -> str:
    """Search memories for a specific query and return formatted results"""
    query = (query or "").strip()
    if not query:
        return "Error: Please provide a search query."
    
    memory = get_memory_system()
    try:
        # Get more results for manual search
        memories = memory.get_relevant_memories(query, limit=5, context_messages=1)
        
        if not memories:
            return f"No memories found for: '{query}'"
        
        # Format the results for Jeeves to read
        result = f"Found {len(memories)} memories for '{query}':\n\n"
        
        for i, memory_group in enumerate(memories, 1):
            short_id = memory_group.get('short_id', '')
            score = memory_group['relevance_score']
            
            result += f"{i}. [ID: {short_id}] (relevance: {score:.2f})\n"
            
            for msg in memory_group['messages']:
                from datetime import datetime
                dt = datetime.fromisoformat(msg['datetime'])
                time_str = dt.strftime("%Y-%m-%d %H:%M")
                
                role = msg['role'].upper()
                if msg.get('agent_id'):
                    role = f"AGENT ({msg['agent_id']})"
                
                content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
                result += f"   [{time_str}] {role}: {content}\n"
            
            result += "\n"
        
        return result.strip()
        
    except Exception as e:
        return f"Error searching memories: {str(e)}"



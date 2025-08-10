#!/usr/bin/env python3
"""
Simple memory system using txtai for Jeeves
Saves all messages, embeds unique ones, retrieves with context
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from txtai.embeddings import Embeddings

class MemorySystem:
    def __init__(self, memory_dir: str = "/Users/daniellosey/JeevesHimself/.memories"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # All messages log file
        self.messages_file = self.memory_dir / "all_messages.jsonl"
        
        # Initialize embeddings lazily to avoid segfault on startup
        self._embeddings = None
        self.embeddings_path = self.memory_dir / "embeddings"
        
        # Initialize index data
        self.embeddings_data = []
        
        # Track embedded messages to avoid duplicates
        self.embedded_messages = self._load_embedded_messages()
        
        # Similarity threshold for considering messages unique
        self.similarity_threshold = 0.85
    
    @property
    def embeddings(self):
        """Lazy initialization of embeddings to avoid startup issues"""
        if self._embeddings is None:
            self._embeddings = Embeddings({
                "path": "sentence-transformers/all-MiniLM-L6-v2",
                "content": True
            })
            
            # Try to load existing embeddings
            try:
                if self.embeddings_path.exists():
                    self._embeddings.load(str(self.embeddings_path))
                    # Reload the data from embedded messages
                    self._reload_embeddings_data()
            except Exception as e:
                print(f"⚠️ Could not load existing embeddings: {e}")
                # If loading fails, we'll start fresh
                
        return self._embeddings
    
    def _load_embedded_messages(self) -> List[str]:
        """Load list of already embedded message IDs"""
        embedded_file = self.memory_dir / "embedded_ids.json"
        if embedded_file.exists():
            with open(embedded_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_embedded_messages(self):
        """Save list of embedded message IDs"""
        with open(self.memory_dir / "embedded_ids.json", 'w') as f:
            json.dump(self.embedded_messages, f)
    
    def _reload_embeddings_data(self):
        """Reload embeddings data from messages file"""
        if not self.messages_file.exists():
            return
        
        with open(self.messages_file, 'r') as f:
            for line in f:
                msg = json.loads(line.strip())
                if msg["id"] in self.embedded_messages:
                    self.embeddings_data.append((msg["id"], msg["content"], None))
    
    def save_message(self, role: str, content: str, agent_id: Optional[str] = None) -> str:
        """Save a message and potentially embed it"""
        # Create message record
        message_id = f"{role}_{int(time.time() * 1000)}"
        message = {
            "id": message_id,
            "role": role,  # "user", "assistant", "agent"
            "content": content,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "agent_id": agent_id
        }
        
        # Always save to messages log
        with open(self.messages_file, 'a') as f:
            f.write(json.dumps(message) + '\n')
        
        # Check if we should embed this message
        if self._should_embed(content):
            # Add to embeddings data
            self.embeddings_data.append((message_id, content, None))
            self.embedded_messages.append(message_id)
            self._save_embedded_messages()
            
            # Rebuild index with all data
            self.embeddings.index(self.embeddings_data)
            
            # Save embeddings periodically
            if len(self.embedded_messages) % 10 == 0:
                self.embeddings.save(str(self.embeddings_path))
        
        return message_id
    
    def _should_embed(self, content: str) -> bool:
        """Check if content is unique enough to embed"""
        if not content.strip():
            return False
        
        # Skip very short messages
        if len(content.strip()) < 10:
            return False
        
        # If no embeddings yet, always embed
        if not self.embedded_messages:
            return True
        
        try:
            # Search for similar messages
            if self.embeddings_data:  # Only search if we have data
                results = self.embeddings.search(content, 1)
                if results and len(results) > 0:
                    score = results[0]["score"]
                    # Only embed if sufficiently different
                    return score < self.similarity_threshold
        except Exception as e:
            print(f"⚠️ Error checking similarity: {e}")
            # If search fails, embed anyway
            return True
        
        return True
    
    def get_relevant_memories(self, query: str, limit: int = 5, context_messages: int = 1) -> List[Dict]:
        """Get relevant memories with surrounding context"""
        if not self.embedded_messages:
            return []
        
        try:
            # Search embeddings
            results = self.embeddings.search(query, limit)
            
            memories = []
            for result in results:
                memory_id = result["id"]
                score = result["score"]
                
                # Get the memory and surrounding messages
                memory_with_context = self._get_message_with_context(memory_id, context_messages)
                if memory_with_context:
                    memories.append({
                        "id": memory_id,
                        "short_id": self._short_id(memory_id),
                        "relevance_score": score,
                        "messages": memory_with_context
                    })
            
            return memories
        except Exception as e:
            print(f"⚠️ Error retrieving memories: {e}")
            return []

    def _short_id(self, message_id: str) -> str:
        """Derive a short, human-friendly id from a message id."""
        if not message_id:
            return ""
        import hashlib
        return hashlib.sha1(message_id.encode()).hexdigest()[:6]
    
    def _get_message_with_context(self, message_id: str, context_size: int = 1) -> List[Dict]:
        """Get a message with N messages before and after for context"""
        all_messages = []
        
        # Load all messages
        with open(self.messages_file, 'r') as f:
            for line in f:
                msg = json.loads(line.strip())
                all_messages.append(msg)
        
        # Find target message
        target_idx = None
        for i, msg in enumerate(all_messages):
            if msg["id"] == message_id:
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        # Get context range
        start_idx = max(0, target_idx - context_size)
        end_idx = min(len(all_messages), target_idx + context_size + 1)
        
        return all_messages[start_idx:end_idx]
    
    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """Format memories for inclusion in system prompt"""
        if not memories:
            return ""
        
        formatted = "\n=== RELEVANT MEMORIES ===\n"
        
        for i, memory_group in enumerate(memories, 1):
            sid = memory_group.get('short_id', '')
            idpart = f" [id: {sid}]" if sid else ""
            formatted += f"\n--- Memory {i}{idpart} (relevance: {memory_group['relevance_score']:.2f}) ---\n"
            
            for msg in memory_group['messages']:
                dt = datetime.fromisoformat(msg['datetime'])
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                
                role = msg['role'].upper()
                if msg.get('agent_id'):
                    role = f"AGENT ({msg['agent_id']})"
                
                formatted += f"[{time_str}] {role}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}\n"
        
        formatted += "\n=== END MEMORIES ===\n"
        return formatted

    def remember(self, text: str) -> str:
        """Create a manual memory note and embed it. Returns short id."""
        note_id = self.save_message("note", text)
        return self._short_id(note_id)

    def forget(self, identifier: str) -> str:
        """Remove a memory from embeddings by id or short id. Keeps raw logs intact."""
        if not identifier:
            return "Error: Provide a memory id to forget."
        full_id = self._resolve_full_id(identifier)
        if not full_id:
            return f"Error: Memory id '{identifier}' not found."
        if full_id in self.embedded_messages:
            self.embedded_messages = [mid for mid in self.embedded_messages if mid != full_id]
            self._save_embedded_messages()
        self.embeddings_data = [t for t in self.embeddings_data if t[0] != full_id]
        try:
            if self.embeddings_data:
                self.embeddings.index(self.embeddings_data)
            else:
                self._embeddings = None
                _ = self.embeddings
                self.embeddings.index([])
            self.embeddings.save(str(self.embeddings_path))
        except Exception as e:
            print(f"⚠️ Error rebuilding embeddings after forget: {e}")
        return f"Forgot memory {self._short_id(full_id)}"

    def _resolve_full_id(self, identifier: str) -> Optional[str]:
        """Find the full message id from a short id or full id among embedded messages."""
        if identifier in self.embedded_messages:
            return identifier
        for mid in self.embedded_messages:
            if self._short_id(mid) == identifier:
                return mid
        return None
    
    def save_conversation_turn(self, user_input: str, assistant_response: str, agent_id: Optional[str] = None):
        """Helper to save a complete conversation turn"""
        self.save_message("user", user_input)
        self.save_message("assistant" if not agent_id else "agent", assistant_response, agent_id)

# Global memory system instance
memory_system = None

def get_memory_system():
    """Get or create the global memory system"""
    global memory_system
    if memory_system is None:
        memory_system = MemorySystem()
    return memory_system

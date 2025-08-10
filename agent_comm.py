#!/usr/bin/env python3
"""
Simple file-based communication system for agent-user questions
"""

import json
import os
import time
from pathlib import Path

COMM_DIR = Path("/Users/daniellosey/JeevesHimself/.agent_comm")
COMM_DIR.mkdir(exist_ok=True)

def post_question(agent_id, question):
    """Post a question from an agent"""
    question_file = COMM_DIR / f"question_{agent_id}_{int(time.time())}.json"
    data = {
        'agent_id': agent_id,
        'question': question,
        'timestamp': time.time(),
        'answered': False,
        'response': None
    }
    with open(question_file, 'w') as f:
        json.dump(data, f)
    return question_file

def check_for_questions():
    """Check for any pending questions"""
    questions = []
    for file in COMM_DIR.glob("question_*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if not data.get('answered', False):
                    questions.append((file, data))
        except:
            pass
    return questions

def answer_question(question_file, answer):
    """Answer a specific question"""
    try:
        with open(question_file, 'r') as f:
            data = json.load(f)
        data['answered'] = True
        data['response'] = answer
        data['answer_time'] = time.time()
        with open(question_file, 'w') as f:
            json.dump(data, f)
        return True
    except:
        return False

def wait_for_answer(question_file, timeout=300):
    """Wait for an answer to a question"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(question_file, 'r') as f:
                data = json.load(f)
                if data.get('answered', False):
                    # Clean up the file
                    os.unlink(question_file)
                    return data['response']
        except:
            pass
        time.sleep(0.5)
    # Timeout - clean up
    try:
        os.unlink(question_file)
    except:
        pass
    return None

def cleanup_old_questions(max_age=3600):
    """Clean up questions older than max_age seconds"""
    now = time.time()
    for file in COMM_DIR.glob("question_*.json"):
        try:
            if file.stat().st_mtime < now - max_age:
                os.unlink(file)
        except:
            pass

#!/usr/bin/env python3
"""Debug script to test memory system initialization"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("1. Testing basic imports...")
try:
    from memory_system import get_memory_system
    print("✓ Memory system import successful")
except Exception as e:
    print(f"✗ Memory system import failed: {e}")
    exit(1)

print("\n2. Testing memory system initialization...")
try:
    memory = get_memory_system()
    print("✓ Memory system initialized")
except Exception as e:
    print(f"✗ Memory system initialization failed: {e}")
    exit(1)

print("\n3. Testing message saving...")
try:
    memory.save_message("user", "Test message")
    print("✓ Message saved successfully")
except Exception as e:
    print(f"✗ Message save failed: {e}")

print("\n4. Testing memory retrieval...")
try:
    memories = memory.get_relevant_memories("test", limit=1)
    print(f"✓ Memory retrieval successful: {len(memories)} memories found")
except Exception as e:
    print(f"✗ Memory retrieval failed: {e}")

print("\n5. Testing txtai directly...")
try:
    from txtai.embeddings import Embeddings
    embeddings = Embeddings({
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "content": True
    })
    print("✓ Txtai embeddings created successfully")
    
    # Try to index some data
    embeddings.index([("1", "test content", None)])
    print("✓ Txtai indexing successful")
    
    # Try to search
    results = embeddings.search("test", 1)
    print(f"✓ Txtai search successful: {len(results)} results")
except Exception as e:
    print(f"✗ Txtai direct test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug complete!")

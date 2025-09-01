# Better than Agents: Smart Python Function Router

### We need smarter tools, not smarter ways of figuring out which tools to use.

This project showcases a smart router for Python functions using HuggingFace transformers for NLP routing and LLMs for intelligent function calling.
Simple, fast, and more efficient than complex agent frameworks.

It doesn't make sense to offload simple routing capabilities to over-powered agents.
Also, letting LLMs focus on the tool they actually need to execute can help lower token consumption and do other cool things.

I'll make the readme better later but I'm lazy and need to use this in real project.

## Key Features

- Smart routing using HuggingFace sentence transformers for fast query classification
- LLM-powered function calling with automatic argument extraction
- Simple decorator-based API for registering functions
- Supports any OpenAI-compatible API (Cerebras, OpenRouter, etc.)
## Why?

Traditional agent frameworks are slow and overcomplicated for most tasks. This router combines:

- **Fast NLP routing**: HuggingFace sentence transformers classify queries to functions
- **Smart argument extraction**: LLMs populate function arguments intelligently  
- **Simple API**: Just decorate your functions and call `smart_call()`

Much faster than complex agent chains while maintaining the flexibility for creative argument generation.

## Usage

```python
import os
from router import Router

# Initialize router with your LLM provider
router = Router(
    api_key=os.getenv("API_KEY"),
    model="gpt-oss-120b", 
    base_url=os.getenv("BASE_URL"),
    provider="Cerebras"
)

# Register functions with example queries
@router.route(potential_queries=[
    "play the song 'Shape of You' by Ed Sheeran", 
    "play the playlist 'Top 100' on Spotify"
])
def play_spotify(song: str = None, playlist: str = None, album: str = None, artist: str = None):
    """Plays music on spotify."""
    if song:
        print(f"Playing {song} on Spotify.")
    elif playlist:
        print(f"Playing {playlist} on Spotify.")
    # ... handle other cases

# Route and execute
router.smart_call("play the song 'Shape of You' by Ed Sheeran")
```

The router automatically:
1. Classifies your query to the best matching function
2. Uses the LLM to extract appropriate arguments 
3. Calls the function with those arguments

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
API_KEY=your_api_key_here
BASE_URL=https://api.cerebras.ai/v1  # or your provider's URL
```

## Requirements

- `openai` - For LLM API calls
- `python-dotenv` - For environment variables  
- `transformers` - For HuggingFace sentence transformers
# Better than Agents: Smart Python Function Router


### We need smarter tools, not smarter ways of figuring out which tools to use.

This project showcases a smart router for Python functions using spaCy for natural language processing and GPT-3.5 for intelligent function calling.
I think this can be better than using many agents for most tasks (but not all!).

It doesn't make sense to use complex (and time consuming) chain-of-thought agents for everything.
# Quick Version:
### Here is how Agents work (according to LangChain)
(source: https://python.langchain.com/docs/use_cases/more/agents/)
![Source LangChain img.png](img.png)
### Here is how Smart Functions can work
![img_1.png](img_1.png)

![img_2.png](img_2.png)

This is a small example, and its aim is mostly to change peoples minds. But hopefully it is also usefull!
## Key Features

- Router class routes text queries to functions based on registered phrases. Uses n-shot huggingface model!
- leverages GPT-3.5 to map queries to function calls   
- Avoid complex virtual agents - route queries directly to functions
- Saves tons of time, you don't want to have to wait 10 seconds for an AI to respond.
## Why?

Many projects try to build virtual assistants or agents that map natural language queries to specific tasks. 
This requires a lot of overhead in training and maintaining complex models. 
Mostly, using LLMs to determine which path to go down, is wastefull and takes time. So much latency!

90% of that can be solved with traditional nlp.
What can't be sovled using traditional nlp is populating the function calls with the correct or creative arguments.
For example, if the user asks for "nice music", nlp can determine to play a song, but it can't determine which song to play.
LLMs can. In this case the query "play some nice music", will quickly be routed to the "play_music(song_name)" function.
Then we can use an LLM to generate a song name like, "Never Gonna Give You Up by Rick Astly"
This is MUCH faster than using OpenAI's function calling or any other agent techniques. If the query is too complex, you can always revert back to agents as a last-case scenario.
This router shows a simpler method:

- Use HuggingFace `zero-shot-classification` pipeline to map queries to functions.  
- Leverage GPT-3.5's few-shot learning for dynamic function calling.

The HuggingFace `deberta` model variant handles basic routing of queries based on registered phrases like "weather in city". 
GPT-3.5 handles actually calling the correct function with the right arguments based on the query.

## Usage

1. Internally, `deberta` routes the query to get_weather() based on registered phrase
2. `SmartFunctionCaller` uses GPT-3.5 to call weather('Paris')
```python
from Smarter.router import SmartRouter
from Smarter.smarter import SmartFunctionCaller

caller = SmartFunctionCaller(<yourapikey>, agent_name="Phoenix")

router = SmartRouter()


@caller.smart_function_call(example_query="what is the weather in the land of lincoln?",
                            example_call="ask_weather('Chicago')") # give it an example
def ask_weather(location: str):
    print("You are asking for weather in:" + location)

```
Our new `ask_weather` function will take a single `query` parameter.
Now we can call:
```python
ask_weather(query='What is the weather the capital of France?')
```
This will output:
```python
"You are asking for the weather in: Paris"
```
## Router Smart Functions (the magic is here)
### First lets define one more smart function
```python
@caller.smart_function_call(example_query="what is happening in Germany right now",
                            example_call="ask_news(topic='Germany', source_url='https://www.bbc.co.uk/')")
def ask_news(topic: str, source_url: str = "https://www.google.com"):
    print(f"getting news on {topic} from {source_url}")
```
Now Lets set up a router to route queries.
```python
from Smarter.router import SmartRouter
router = SmartRouter()
@router.route(["what is the weather in location"]) # example queries for the deberta model
def get_weather_in_city(text):
    ask_weather(query=text) # our old weather function

@router.route(["asking for news", "current events"])
def get_news_category(text):
    ask_news(query=text)
```
Now we can ask our router to route a query to the correct function by simply calling:
```python
router.query_and_call("Tell me the news about...")
```

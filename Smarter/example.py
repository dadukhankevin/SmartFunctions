from Smarter.router import SmartRouter
from Smarter.smarter import SmartFunctionCaller

caller = SmartFunctionCaller("your_anthropic_api_key", agent_name="Phoenix")

router = SmartRouter()


@caller.smart_function_call(example_query="what is the weather in the land of lincoln?",
                            example_call="ask_weather('Chicago')")
def ask_weather(location: str):
    print("You are asking for weather in:" + location)


@caller.smart_function_call(example_query="what is happening in Germany right now",
                            example_call="ask_news(topic='Germany', source_url='https://www.bbc.co.uk/')")
def ask_news(topic: str, source_url: str = "https://www.google.com"):
    """Find news on a specific topic from a specific source, default CNN"""
    print(f"getting news on {topic} from {source_url}")

print(ask_news("what is the news in Germany"))

@router.route(["what is the weather in location"])
def get_weather_in_city(text):
    r = ask_weather(query=text)
    print(f"Getting weather in {r}")
    return r


@router.route(["asking for news", "current events"])
def get_news_category(text):
    r = ask_news(query=text)
    return r


result = router.query_and_call("What's the news")
print(result)
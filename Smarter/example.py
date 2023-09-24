from Smarter.router import Router
from Smarter.smarter import smart_function

router = Router()

@smart_function
def ask_weather(location: str):
    """The location you are asking the weather of"""
    #tbd
    return location

@smart_function
def ask_news(topic: str, source_url: str):
    """Find news on a specific topic from a specific source, default CNN"""
    #tbd
    return f"getting news on {topic} from {source_url}"

@router.route(["weather in city"])
def get_weather_in_city(text):
    r = ask_weather(query=text)
    return r

@router.route(["news about category"])
def get_news_category(text):
    r = ask_news(query=text)
    return r
result = router.query_and_call("What's the weather in San Francisco? Also show me tech news.")
print(result)
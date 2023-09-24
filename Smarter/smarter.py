
import openai
import functools
import inspect
import re
import random

openai.api_key = input("Enter your OpenAI api key: ")


class spotify:
    def __init__(self):
        pass
    def play(self, song):
        pass


def smart_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query = kwargs.pop("query", None)
        if query:
            model = "gpt-3.5-turbo"
            function_signature = inspect.getsource(func).split("\n")[1:-1]

            messages = [
                {"role": "system",
                 "content": f"Random seed: {random.randint(0,1000)}. Generate code that calls {function_signature}. Stick to the format: {func.__name__}(arg1, arg2, ...)."
                            f"If the user asks to 'check the weather in a sunny location' and the function is 'def check_weather(location)'"
                            f"simply return ```check_weather('Hawaii')"},
                {"role": "user", "content": query}
            ]

            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=1)
            assistant_message = response['choices'][0]['message']['content']
            print(assistant_message)

            try:
                arg_str = re.search(f"{func.__name__}\((.*?)\)", assistant_message).group(1)
                arg_list = [arg.strip().replace("'","").replace('"',"") for arg in arg_str.split(',')]
            except AttributeError:
                return f"Could not extract arguments for {func.__name__} based on query."

            return func(*arg_list)
        else:
            return func(*args, **kwargs)

    return wrapper



import openai
import functools
import inspect
import re
import random

openai.api_key = input("Please enter your OpenAI api key:\n>>>")


class spotify:
    def __init__(self):
        pass
    def play(self, song):
        pass

def extract(func_call, **kwargs):
    match = re.match(r"(\w+)\((.*?)\)", func_call)
    if match:
        args = match.group(2)
        return args, kwargs
    else:
        return None, kwargs

def smart_function(func):
    @functools.wraps(func)
    def wrapper(query):
        if query:
            model = "gpt-3.5-turbo"
            function_signature = inspect.getsource(func).split("\n")[0:-1]
            function_signature = "```\n".join(function_signature[1:min(5,len(function_signature)-1)]) + "\n\t...```"
            print(function_signature)
            messages = [
                {"role": "system",
                 "content": f"You intelligently call functions here are some examples: "
                            f"function: play_song(song_name)"
                            f"query: play an underrated Ed Sheeran song"
                            f"function call: play_song('Nancy Mulligan by Ed Sheeran', type='tracks')"
                            f"function: check_weather(location)"
                            f"query: whats it like temperature wise in the state where everything is big?"
                            f"function_call: check_weather('Dallas, Texas')"
                            f"etc..."},


                {"role": "user", "content": f"function: ```{function_signature}```\n"
                                            "query: '" + query +
                                            f"\nfunction call: {func.__name__}(<fill this in>) respond with just the one line, `function(args)`!",}

            ]

            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=.6)
            assistant_message = response['choices'][0]['message']['content'].replace("function call: ","").replace("`","")
            assistant_message = func.__name__+assistant_message.split(func.__name__)[-1]
            print(assistant_message)

        exec(assistant_message.replace(f"{func.__name__}(","func("))

    return wrapper


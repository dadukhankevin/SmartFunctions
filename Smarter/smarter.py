import openai
import functools
import re

def extract(func_call, **kwargs):
    match = re.match(r"(\w+)\((.*?)\)", func_call)
    if match:
        args = match.group(2)
        return args, kwargs
    else:
        return None, kwargs


def find_function_calls(text):
    # Regex pattern to find function calls
    pattern = r'\w+\([^\)]*\)'
    return re.findall(pattern, text)


class SmartFunctionCaller:
    def __init__(self, api_key, agent_name="Phoenix"):
        self.prompt = """You are {agent_name}. You call a function intelligently using the users query. For example:
        user_query: {example_query}
        text_function_call: {example_call}
        Now, create the '{function_name}' function call using this user_query: {user_query}"""
        openai.api_key = api_key
        self.agent_name = agent_name
        self.model = "gpt-3.5-turbo"
        self.function_names = []

    def smart_function_call(self, example_query, example_call):
        def decorator(func):
            self.function_names.append(func.__name__)

            @functools.wraps(func)
            def wrapper(query):
                whole_prompt = self.prompt.format(agent_name=self.agent_name, example_query=example_query,
                                                  example_call=example_call, user_query=query,
                                                  function_name=func.__name__)

                messages = [{"role": "system", "content": whole_prompt}, {"role": "user",
                                                                          "content": "Create the function call based on the query"}]

                response = openai.ChatCompletion.create(model=self.model, messages=messages, temperature=.6)
                assistant_message = response['choices'][0]['message']['content']
                function_calls = find_function_calls(assistant_message)
                for function_call in function_calls:
                    exec(function_call.replace(func.__name__ + "(","func("))  # Assign the return value to result
            return wrapper

        return decorator


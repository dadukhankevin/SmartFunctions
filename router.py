import inspect
from openai import OpenAI
from typing import Callable
from transformers import pipeline

_pipeline = pipeline("zero-shot-classification", model='sentence-transformers/all-MiniLM-L6-v2')

SYSTEM_PROMPT = """
You will be given a single tool, and a text query.
Your job is to call the tool in a way that maximially satisfies the query.
The tools in question are really just python functions! So to call them, you must simply use this format:
<python>
some_tool(arg1, arg2=something, ...)
</python>
You determine the arguments to pass etc, and when possible you may be given more context to help besides just the query.
Do not forget the python tag, and do not include any non-python code within it.
"""

class Router:
    def __init__(self, api_key: str, model: str, base_url: str, provider: str):
        self.labels: dict[str, Callable] = {}
        self.classifier = _pipeline
        self.global_context = ""

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.provider = provider
        self.model = model
    def _extract_outermost_enclosed_text(self, text: str):
        start = text.find('(')
        end = text.rfind(')')
        return text[start + 1:end]

    def _smart_call(self, function: Callable, promptable_code: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": promptable_code}
            ],
            extra_body={
                "provider": {
                    "order": [self.provider],
                    "allow_fallbacks": True
                }
            }
        )
        completion = str(response.choices[0].message.content)
        python_code = completion.split("<python>")[1].split("</python>")[0].strip()
        # Extract function call and execute it
        return eval(python_code, {function.__name__: function})


    def route(self, potential_queries: list[str]):
        def decorator(func):
            def wrapper(query: str):
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or ""
              
                rets = []
                code = inspect.getsource(func)

                for line in code.splitlines():
                    line = line.strip()
                    if "return " in line:
                        rets.append(line)

                promptable_code = (
                    f"Some general context that may be useful: {self.global_context}\n"
                    f"Function schema:\n"
                    f"def {func.__name__}{sig}:\n"
                    f'    """\n\t{doc.replace(chr(10), " ")}\n\t"""\n'
                    f"    # Function code here.\n"
                    f"All return statements:\n\t" + "\n\t".join(rets)
                )
                promptable_code += f"\nPlease call this function in a way that maximially satisfies the query: \n'{query}'"
                return self._smart_call(function=func, promptable_code=promptable_code)
            for query in potential_queries:
                self.labels[query] = wrapper
            


        return decorator

    def smart_call(self, text):
        scores = self.classifier(text, list(self.labels.keys()))
        best_keyphrase = scores['labels'][0]
        best_func = self.labels.get(best_keyphrase)

        if best_func:
            return best_func(text)
        else:
            return None

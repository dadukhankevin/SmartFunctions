import spacy

class Router:
    def __init__(self):
        self.functions = {}
        self.requirements = {}
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize_and_normalize(self, text):
        doc = self.nlp(text.lower())
        return set([token.lemma_ for token in doc if token.is_alpha])  # Removed punctuations

    def route(self, uris: list):
        def wrapper(func):
            for uri in uris:
                norm_uri = self.tokenize_and_normalize(uri)
                self.requirements[frozenset(norm_uri)] = func.__name__
            self.functions[func.__name__] = func
            return func
        return wrapper

    def query(self, text):
        norm_text = self.tokenize_and_normalize(text)
        results = []
        for uri, func_name in self.requirements.items():
            if uri & norm_text:  # Checks for at least one common element
                results.append(self.functions[func_name])
        return results

    def query_and_call(self, text):
        functions = self.query(text)
        return [{func.__name__: func(text)} for func in functions]

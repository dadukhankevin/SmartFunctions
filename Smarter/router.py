from transformers import pipeline

_pipeline = pipeline("zero-shot-classification", model='cross-encoder/nli-deberta-base')


class SmartRouter:
    def __init__(self):
        self.labels = {}
        self.classifier = _pipeline

    def route(self, keyphrases: list):
        def wrapper(func):
            for keyphrase in keyphrases:
                self.labels.update({keyphrase: func})
            return func

        return wrapper

    #
    def query_and_call(self, text):
        scores = self.classifier(text, list(self.labels.keys()))
        best_keyphrase = scores['labels'][0]
        best_func = self.labels.get(best_keyphrase)

        if best_func:
            return best_func(text)
        else:
            return

from difflib import SequenceMatcher
import re
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")


def remove_brackets(text):
    return re.sub(r'\[[^\]]*\]', '', text)

def normalized_similarity(text1, text2):
    s = SequenceMatcher(None, text1, text2)
    raw_ratio = s.ratio()

    # Find the length of the longer text
    max_len = max(len(text1), len(text2))

    # Normalize the raw_ratio
    normalized_ratio = raw_ratio * (min(len(text1), len(text2)) / max_len)

    return normalized_ratio

def extract_bracketed_words(text):
    return re.findall(r'\[([^\]]*)\]', text)


def automatic_threshold_sort(input_list):
    element_count = Counter(input_list)
    sorted_elements = sorted(element_count.items(), key=lambda x: x[1], reverse=True)
    total_count = len(input_list)
    mean_threshold = total_count // 2
    filtered_list = [item[0] for item in sorted_elements if item[1] >= mean_threshold]
    return list(set(filtered_list))


class Router:
    def __init__(self):
        self.functions = {}
        self.requirements = {}
        self.bracketed_POS = {}
        self.weights = {}

    def route(self, keywords: list):
        def wrapper(func):
            keyword_str = " ".join(keywords)
            doc = nlp(keyword_str)
            pos_tags = {token.text: token.pos_ for token in doc if token.text in extract_bracketed_words(keyword_str)}
            self.functions[func.__name__] = func
            self.requirements[func.__name__] = keyword_str
            self.bracketed_POS[func.__name__] = pos_tags

            # Store weights
            self.weights[func.__name__] = {}
            for keyword in keywords:
                weight = keyword.count("*")
                self.weights[func.__name__][keyword.lstrip('*')] = weight + 1

            return func

        return wrapper

    def query(self, text: str, thresh=.4):
        text = text.lower()
        results = []
        text_doc = nlp(text)
        text_pos = {token.text: token.pos_ for token in text_doc}

        for func_name, keyword_str in self.requirements.items():
            base_similarity = normalized_similarity(remove_brackets(text), remove_brackets(keyword_str))
            bonus = 0

            for word, weight in self.weights[func_name].items():
                if word.lower() in text.lower():
                    bonus += (0.1 * weight)

            for word, pos in self.bracketed_POS[func_name].items():
                if pos in text_pos.values():
                    bonus += 0.1

            final_similarity = base_similarity + bonus
            if final_similarity > thresh:
                results.append((func_name, self.functions[func_name], final_similarity))

        results = sorted(results, key=lambda x: x[2], reverse=True)
        print(results)
        threshold = thresh
        best_results = [result for result in results if result[2] >= threshold]
        print(best_results)
        best_results = [result[1] for result in best_results]

        return best_results

    def query_and_call(self, text: str, thresh=.4):
        functions = list(set(self.query(text, thresh)))
        return [{func.__name__: func(text)} for func in functions]
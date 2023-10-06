"""
Nome: Leonardo Saponi de Souza
Data: 10/2023
"""
import nltk
import re
import sys
from collections import defaultdict
from nltk.stem import RSLPStemmer

nltk.download('stopwords')
nltk.download('rslp')

stop_words = set(nltk.corpus.stopwords.words("portuguese"))
stemmer = RSLPStemmer()

def build_inverted_index(file_paths, stop_words):
    inverted_index = defaultdict(list)
    for doc_id, path in enumerate(file_paths, start=1):
        try:
            with open(path.strip(), encoding='utf8') as file:
                content = file.read()
                words = nltk.word_tokenize(content)
                clean_words = [re.sub(r'[, .!?...]', '', word) for word in words if word.strip()]
                filtered_words = [
                    stemmer.stem(word.lower()) for word in clean_words 
                    if word and word.lower() not in stop_words
                ]
                for word in filtered_words:
                    if word in inverted_index and inverted_index[word][-1][0] == doc_id:
                        inverted_index[word][-1] = (doc_id, inverted_index[word][-1][1] + 1)
                    else:
                        inverted_index[word].append((doc_id, 1))
        except FileNotFoundError:
            print(f"Arquivo {path} não encontrado.")
    return inverted_index

def solve_query(query, inverted_index, num_docs):
    terms = query.split()
    documents = set(range(1, num_docs + 1))
    if len(terms) == 1:
        term = stemmer.stem(terms[0])
        if term.startswith("!"):
            return documents.difference({doc_id for doc_id, _ in inverted_index.get(term[1:], [])})
        else:
            return {doc_id for doc_id, _ in inverted_index.get(term, [])}
    for i, term in enumerate(terms):
        if term.startswith("!"):
            terms[i] = documents.difference({doc_id for doc_id, _ in inverted_index.get(term[1:], [])})
    new_query = []
    temp = []
    for term in terms:
        if term == "&":
            continue
        elif term == "|":
            new_query.append(temp)
            temp = []
        else:
            if isinstance(term, set):
                temp.append(term)
            else:
                term = stemmer.stem(term)
                temp.append({doc_id for doc_id, _ in inverted_index.get(term, [])})
    if temp:
        new_query.append(temp)
    final_results = set()
    for group in new_query:
        intersection = set(range(1, num_docs + 1))
        for term in group:
            intersection = intersection.intersection(term)
        final_results = final_results.union(intersection)
    return final_results

base_file_path = sys.argv[1]
query_file_path = sys.argv[2]

try:
    with open(base_file_path, encoding='utf8') as base_file:
        arrayPaths = base_file.readlines()
except FileNotFoundError:
    print(f"Arquivo base {base_file_path} não encontrado.")

try:
    with open(query_file_path, encoding='utf8') as query_file:
        query = query_file.read()
except FileNotFoundError:
    print(f"Arquivo de consulta {query_file_path} não encontrado.")

inverted_index = build_inverted_index(arrayPaths, stop_words)
final_results = solve_query(query, inverted_index, len(arrayPaths))

with open('indice.txt', 'w', encoding='utf8') as index_file:
    for word, occurrences in inverted_index.items():
        line = f'{word}: '
        for doc_id, count in occurrences:
            line += f'{doc_id},{count} '
        index_file.write(line.strip() + '\n')

with open('resposta.txt', 'w', encoding='utf8') as answer_file:
    answer_file.write(f'{len(final_results)}\n')
    for doc_id in final_results:
        answer_file.write(f'{arrayPaths[doc_id-1]}')
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Load the knowledge base
with open("../data/kb.json", "r") as file:
    knowledge_base = json.load(file)
 
# Prepare questions for TF-IDF
questions = [entry['question'] for entry in knowledge_base]
 
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)
 
def retrieve_answer(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()
    return knowledge_base[top_index]['answer']
 
# Test the retriever
user_query = "Tell me about RAG"
answer = retrieve_answer(user_query)
print("Retrieved Answer:", answer)
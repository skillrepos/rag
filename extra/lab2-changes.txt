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

import requests

# Use Ollama to generate a detailed response based on the retrieved answer
ollama_url = "http://localhost:11434/api/generate"
 
query = {
    "model": "llama3",
    "prompt": f"Expand on the following answer:\n\n{answer}",
    "stream": False
}
 
response = requests.post(ollama_url, json=query)
 
# Check if the request was successful
if response.status_code == 200:
    # Process the response
    if query["stream"]:
        # Handle streaming response
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                print(body.get("response", ""))
    else:
        # Handle non-streaming response
        response_data = response.json()
        print(response_data["response"])
else:
    # Print error message
    print("Error:", response.status_code, response.text)

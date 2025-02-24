# Step 1: Import libraries and prepare RAG training examples.
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

 
# Open and read the JSON file
json_file_path = 'data/small_rag.json'
with open(json_file_path, 'r') as file:
    rag_examples = json.load(file)

login(token="Token Enter here")


def retrieve_context(new_query, top_k=5):
    """
    Given a new query, compute its TF-IDF vector, then retrieve the top_k similar examples.
    """
    new_query_vector = vectorizer.transform([new_query])
    # Compute cosine similarities (dot product on L2-normalized vectors)
    similarities = np.dot(query_vectors, new_query_vector.T).toarray().flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_examples = [rag_examples[i] for i in top_indices]
    return retrieved_examples

#Build a prompt by incorporating retrieved examples and the new query.
def build_prompt(new_query, retrieved_examples,data):
    prompt = (
        "You are a helpful assistant that generates Python pandas code based on a natural language query.\n"
        "Below are some examples:\n\n"
    )
    for ex in retrieved_examples:
        prompt += f"Query: {ex['query']}\n"
        prompt += f"Context: {ex['retrieved_context']}\n"
        prompt += f"Code: {ex['generated_code']}\n\n"
    prompt += f"New Query: {new_query}\n"
    prompt += "Generate the corresponding Python pandas code."
    prompt += "Generate only the most relevant python pandas code, nothing more"
    prompt += f"The data consist of the following columns {data.columns.tolist()}"
    return prompt

def extract_insight(generated_text,data):
  generated_code = generated_text.split('Code:')[-1].split("\n")[0].strip()
  local_vars = {"df": data}
  # Execute the generated code safely
  exec(generated_code, {}, local_vars)
  result = local_vars["result"]
  return result

def LLM(new_query,df):
    # Build a TF-IDF vectorizer using the queries from the RAG examples.
    queries = [ex['query'] for ex in rag_examples]
    vectorizer = TfidfVectorizer().fit(queries)
    query_vectors = vectorizer.transform(queries)
    #Load the Mistral7B model and its tokenizer.
    model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # retrieve examples, build the prompt, and generate code based on the user prompt
    retrieved_examples = retrieve_context(new_query)
    prompt = build_prompt(new_query, retrieved_examples,df)
    # Encode the prompt for the model.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate output from the model.
    output = model.generate(
        **inputs,
        max_new_tokens=150,# Adjust as needed.
        do_sample=True,
        temperature=0.7
    )
    # Decode the generated text.
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    result = extract_insight(generated_text,df)
    #json_str = df.to_json(orient='records') 
    return result

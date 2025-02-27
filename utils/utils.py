# Step 1: Import libraries and prepare RAG training examples.
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from dotenv import load_dotenv
import os

 
# Open and read the JSON file
json_file_path = 'data/small_rag.json'
with open(json_file_path, 'r') as file:
    rag_examples = json.load(file)


def retrieve_context(new_query, top_k=3):
    """
    Retrieve the top_k similar examples based on cosine similarity of TF-IDF vectors.

    This function builds a TF-IDF vectorizer using the 'query' field from the 
    global RAG examples list, transforms the new query into its TF-IDF vector, 
    and computes cosine similarity scores with all stored queries. The top_k most 
    similar examples are returned.

    Parameters:
        new_query (str): The new query string for which similar examples are sought.
        top_k (int): The number of top similar examples to retrieve (default is 3).

    Returns:
        list: A list of the top_k examples (dictionaries) most similar to the new query.

    Raises:
        ValueError: If 'new_query' is not a non-empty string.
        RuntimeError: If the global 'rag_examples' list is not defined or is empty.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Check if rag_examples exists and is non-empty.
    try:
        if not rag_examples:
            raise RuntimeError("The global 'rag_examples' list is not defined or is empty.")
    except NameError:
        raise RuntimeError("The global 'rag_examples' list is not defined.")

    # Validate that new_query is a non-empty string.
    if not isinstance(new_query, str) or not new_query.strip():
        raise ValueError("The 'new_query' parameter must be a non-empty string.")

    # Extract the list of queries from the rag_examples.
    queries = [ex['query'] for ex in rag_examples if 'query' in ex]
    if not queries:
        raise RuntimeError("No valid 'query' entries found in 'rag_examples'.")

    # Build a TF-IDF vectorizer using the queries from the RAG examples.
    vectorizer = TfidfVectorizer().fit(queries)
    query_vectors = vectorizer.transform(queries)

    # Transform the new query to its TF-IDF vector.
    new_query_vector = vectorizer.transform([new_query])

    # Compute cosine similarities (dot product on L2-normalized vectors).
    similarities = np.dot(query_vectors, new_query_vector.T).toarray().flatten()

    # Get indices of the top_k most similar examples.
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve the corresponding examples from rag_examples.
    retrieved_examples = [rag_examples[i] for i in top_indices]

    return retrieved_examples

def build_prompt(new_query, retrieved_examples, data):
    """
    Build a prompt by incorporating retrieved examples and the new query.

    This function constructs a prompt for a Python pandas code generation assistant.
    It includes a set of example queries with their corresponding context and generated code,
    followed by the new query and the data's column information.

    Parameters:
        new_query (str): The new natural language query.
        retrieved_examples (list): A list of dictionaries, each containing keys:
            'query', 'retrieved_context', and 'generated_code'.
        data (pandas.DataFrame): A DataFrame whose columns will be included in the prompt.

    Returns:
        str: A prompt string constructed for the code generation task.

    Raises:
        ValueError: If new_query is not a non-empty string.
        ValueError: If retrieved_examples is not a list or is empty.
        ValueError: If data is not a pandas DataFrame.
    """
    # Validate new_query is a non-empty string.
    if not isinstance(new_query, str) or not new_query.strip():
        raise ValueError("new_query must be a non-empty string.")

    # Validate retrieved_examples is a non-empty list.
    if not isinstance(retrieved_examples, list) or not retrieved_examples:
        raise ValueError("retrieved_examples must be a non-empty list.")

    # Validate data is a pandas DataFrame.
    try:
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
    except ImportError:
        raise ImportError("pandas must be installed to use this function.")

    # Initialize the prompt with introductory context.
    prompt = (
        "You are a helpful assistant that generates Python pandas code based on a natural language query.\n"
        "Below are some examples:\n\n"
    )

    # Append each retrieved example to the prompt.
    for ex in retrieved_examples:
        # Check if each expected key exists in the example.
        if not all(key in ex for key in ['query', 'retrieved_context', 'generated_code']):
            raise ValueError("Each example in retrieved_examples must contain 'query', 'retrieved_context', and 'generated_code' keys.")

        prompt += f"Query: {ex['query']}\n"
        prompt += f"Context: {ex['retrieved_context']}\n"
        prompt += f"Code: {ex['generated_code']}\n\n"

    prompt += "End of examples\n\n"

    # Append the new query.
    prompt += f"\n\nNew Query: {new_query}\n"

    # Append instructions for generating the Python pandas code.
    prompt += "Generate the Python pandas code and return as a json ."

    prompt += "If the query is for creating a new column do it inplacef\n"
    prompt += "Include the datatype changes in the code as required"

    # Append the data's column information.
    prompt += f"The data consist of the following columns {data.columns.tolist()}"

    prompt += "If the task involves creating or modifying columns (for example, calculating new values), modify the existing DataFrame by adding the new columns directly to it."

    prompt +="\n For any other operations (such as filtering, aggregations, merging, etc.), assign the output to a variable named result."
    return prompt



def extract_insight(generated_text, data):
    """
    Extracts and executes Python pandas code from the generated text to derive insights from the given DataFrame.

    This function extracts the generated Python pandas code from the given text, executes it in a local scope, 
    and returns the resulting output.

    Parameters:
        generated_text (str): The generated text containing Python pandas code.
        data (pandas.DataFrame): The DataFrame (df) on which the extracted code will be executed.

    Returns:
        object: The result of executing the extracted pandas code.

    Raises:
        ValueError: If generated_text is not a valid string or does not contain extractable code.
        ValueError: If data is not a pandas DataFrame.
        KeyError: If the executed code does not produce a variable named 'result'.
        SyntaxError: If the extracted code is not valid Python.
        Exception: If any other error occurs during code execution.
    """
    import pandas as pd

    # Validate that generated_text is a non-empty string.
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise ValueError("generated_text must be a non-empty string.")

    # Validate that data is a pandas DataFrame.
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    try:
        # Extract the generated code from the text.
        if "Code:" not in generated_text:
          return ""
        
        # Split the text at 'Code:' and take the portion after it.
        code_section = generated_text.split("Code:")[-1]
        # The code might be on the first line; we split by newline and strip extra whitespace.
        extracted_code = code_section.split("\n")[0].strip()


        if not extracted_code:
            raise ValueError("No valid code found in generated_text.")

        # Initialize a local execution environment with 'df' as the DataFrame.
        local_vars = {"df": data}

        # Execute the extracted code safely.
        exec(extracted_code, {}, local_vars)

        # Ensure the 'result' variable is defined in the executed code.
        if "result" not in local_vars:
            raise KeyError("The executed code did not produce a variable named 'result'.")

        return local_vars["result"]

    except SyntaxError:
        raise SyntaxError("The extracted code contains syntax errors.")
    except Exception as e:
        raise Exception(f"An error occurred while executing the generated code: {str(e)}")


def LLM(new_query, df, rag_examples):
    """
    Generate insights using a large language model with retrieval-augmented examples.

    This function leverages an LLM (specifically the Mistral-7B model) to generate Python pandas code based on a natural language query.
    It retrieves similar examples from a provided list of RAG examples, constructs a prompt that includes these examples and data details,
    generates the code using the model, executes the generated code on the provided DataFrame, and returns the resulting insight.

    Parameters:
        new_query (str): The natural language query to generate the pandas code.
        df (pandas.DataFrame): The DataFrame on which the generated code will operate.
        rag_examples (list): A list of dictionaries, each containing keys 'query', 'retrieved_context', and 'generated_code'.

    Returns:
        object: The result obtained from executing the generated code on the DataFrame.

    Raises:
        ValueError: If new_query is not a non-empty string.
        ValueError: If df is not a pandas DataFrame.
        ValueError: If rag_examples is not a non-empty list.
        Exception: If an error occurs during model loading, prompt construction, code generation, or code execution.
    """
    try:
        # Validate inputs.
        if not isinstance(new_query, str) or not new_query.strip():
            raise ValueError("new_query must be a non-empty string.")
        
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this function.")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not isinstance(rag_examples, list) or not rag_examples:
            raise ValueError("rag_examples must be a non-empty list.")

        # Load the Mistral-7B model and its tokenizer.
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Retrieve similar examples from rag_examples using the new_query.
        # (Assuming retrieve_context is defined to use the global rag_examples or has been updated accordingly.)
        retrieved_examples = retrieve_context(new_query)
        
        # Build the prompt using the retrieved examples, the new query, and data details.
        prompt = build_prompt(new_query, retrieved_examples, df)

        # Encode the prompt for the model.
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output from the model.
        output = model.generate(
            **inputs,
            max_new_tokens=500,  # Adjust the number of tokens as needed.
            do_sample=True,
            temperature=0.7
        )

        # Decode the generated text.
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text to isolate the new content.
        generated_text_trim = generated_text.replace(prompt, "")

        print(f"generated_text_trim:{generated_text_trim}")
        
        # Extract and execute the generated code to obtain the result.
        result = extract_insight(generated_text_trim, df)

        try:
            # Convert DataFrame to JSON string with 'records' orientation.
            json_result = result.to_json(orient='records')
            print("JSON output:", json_result)
        except Exception as e:
            print("Error converting DataFrame to JSON:", e)

        return json_result

    except Exception as e:
        raise Exception(f"An error occurred in the LLM function: {str(e)}")

def chat_with_groq(client, prompt, model, response_format):
  completion = client.chat.completions.create(
  model=model,
  messages=[
      {
          "role": "user",
          "content": prompt
      }
  ],
  response_format=response_format
  )

  return completion.choices[0].message.content

def LLM_GROQ(new_query, df,file_path):
    """
    Generate insights using a large language model with retrieval-augmented examples.

    This function leverages an LLM (specifically the Mistral-7B model) to generate Python pandas code based on a natural language query.
    It retrieves similar examples from a provided list of RAG examples, constructs a prompt that includes these examples and data details,
    generates the code using the model, executes the generated code on the provided DataFrame, and returns the resulting insight.

    Parameters:
        new_query (str): The natural language query to generate the pandas code.
        df (pandas.DataFrame): The DataFrame on which the generated code will operate.
        rag_examples (list): A list of dictionaries, each containing keys 'query', 'retrieved_context', and 'generated_code'.

    Returns:
        object: The result obtained from executing the generated code on the DataFrame.

    Raises:
        ValueError: If new_query is not a non-empty string.
        ValueError: If df is not a pandas DataFrame.
        ValueError: If rag_examples is not a non-empty list.
        Exception: If an error occurs during model loading, prompt construction, code generation, or code execution.
    """
    try:
        # Validate inputs.
        if not isinstance(new_query, str) or not new_query.strip():
            raise ValueError("new_query must be a non-empty string.")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this function.")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not isinstance(rag_examples, list) or not rag_examples:
            raise ValueError("rag_examples must be a non-empty list.")

        # Retrieve similar examples from rag_examples using the new_query.
        # (Assuming retrieve_context is defined to use the global rag_examples or has been updated accordingly.)
        retrieved_examples = retrieve_context(new_query)

        # Build the prompt using the retrieved examples, the new query, and data details.
        prompt = build_prompt(new_query, retrieved_examples, df)
        # Use the Llama3 70b model
        model = "llama3-70b-8192"
        # Get the Groq API key and create a Groq client
        #groq_api_key = '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'
        # Load variables from .env file
        load_dotenv(".env")
        groq_api_key = os.getenv("groq_api_key")
        client = Groq(
          api_key=groq_api_key
        )
        # Get the AI's response. Call with '{"type": "json_object"}' to use JSON mode
        llm_response = chat_with_groq(client, prompt, model, {"type": "json_object"})
        result_json = json.loads(llm_response)
        if 'code' in result_json:
          pandas_query = result_json['code']
                  
          # Initialize a local execution environment with 'df' as the DataFrame.
          local_vars = {"df": df}
          pandas_query = f"""import pandas as pd\nimport numpy as np\n{pandas_query}
          """
          print(pandas_query)  

          # Execute the extracted code safely.
          exec(pandas_query, {}, local_vars)
        try:
            # Convert DataFrame to JSON string with 'records' orientation.
            if "result" in local_vars:
                json_result = local_vars['result'].to_json(orient='records')
            else:
                local_vars['df'].to_csv(file_path)
                json_result = {"Dataframe succcesfully updated!"}
        except Exception as e:
            print("Error converting DataFrame to JSON:", e)
        return json_result

    except Exception as e:
        raise Exception(f"An error occurred in the LLM function: {str(e)}")

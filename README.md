# Text-to-Pandas

A command line application that allows users to ask questions about their CSV data: the application leverages Groq API's JSON mode to generate Pandas codes based on the user's queries in natural language and execute them on a Pandas data frame.


## Features

* Text-to-Pandas: The application uses natural language processing to convert user questions into Pandas codes, making it easy for users to query their data without knowing Python or Data Analytics.

* JSON mode: A feature which enables the LLM to respond strictly in a structured JSON output, provided the desired format is given

## Data

* Data in CSV format can be uploaded and used for analysis

## Usage

* You must store a valid Groq API Key as a secret to proceed with this example.
* Update the Key in .env file
* run it on the command line with 
```bash
run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
OR 
```bash
python main.py
```

## Docs
* The API doc is available at http://localhost:8000/docs

## Approach

* Created a synthetic dataset for eCommerce Transactions
* Created a knowledge base for the Query to Pandas code conversion including query, retrieved_context and generated_code
* A TF-IDF-based context retrieval is implemented (Tested FAISS and realized TF-IDF Approach is better - added notebook for reference)
* Prompt Engineering is Done for optimization 
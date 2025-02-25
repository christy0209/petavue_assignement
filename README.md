Steps to Run the application

Option 1 - Execute the Dockerfile

Option 2 - Run locally

*    Update the HF token
*    run uvicorn main:app --reload --host 0.0.0.0 --port 8000
*    Access the Swagger doc at http://localhost:8000/docs
*    Also see the experiment under notebooks folder

Approach

* Created a synthetic dataset for eCommerce Transactions
* Cretatd a knowledge base for the Query to Pandas code conversion including query, retrieved_context and generated_code
* A TF-IDF based context retrival is implemented (Tested FAISS and realized TF-IDF Approach is beter - added notebook for reference)
* Prompt Enginnering is Done for optimization 
* A small funciton (extract_insight) is create for executing the pandas code and to return the dataframe

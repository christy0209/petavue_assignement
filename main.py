from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
import os
import shutil
import logging
from groq import Groq
from utils.utils import LLM_GROQ

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CSV File Processing API",
    description="API for uploading CSV files, processing them, and interacting with the data via an LLM.",
    version="1.0.0"
)

# Allowed content types for CSV files
ALLOWED_CONTENT_TYPES = [
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel"
]


@app.post("/upload/", summary="Upload an CSV File", tags=["File Operations"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload an CSV file.

    **File Upload Requirements:**
    - File extension should be `.csv`

    On success, the file is saved in the `uploads` directory.
    """
    logger.info("Received file upload request for file: %s", file.filename)

    # Verify the content type of the uploaded file
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.error("Invalid file type: %s", file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an CSV file.")

    filename = file.filename
    if not (filename.endswith(".csv")):
        logger.error("Invalid file extension for file: %s", filename)
        raise HTTPException(status_code=400, detail="File extension is not allowed. Please upload an CSV file.")

    # Save the file to disk in the uploads directory
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    logger.info("Saving file to: %s", save_path)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved successfully: %s", filename)
    except Exception as e:
        logger.error("Error saving file %s: %s", filename, e)
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()

    return JSONResponse(content={"message": "File uploaded and saved successfully", "filename": filename})

@app.delete("/delete/", summary="Delete an Uploaded File", tags=["File Operations"])
async def delete_file(file_name: str = Query("transactions.csv", description="Name of the file to delete from the uploads directory")):
    """
    Delete an uploaded CSV file from the server.
    
    **Parameters:**
    - **file_name**: Name of the file to delete (default: `transactions.csv`).
    
    Returns a success message if the file is deleted, or an error if the file does not exist.
    """
    file_path = os.path.join('uploads', file_name)
    logger.info("Request to delete file: %s", file_path)
    
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise HTTPException(status_code=404, detail="File not found.")
    
    try:
        os.remove(file_path)
        logger.info("File deleted successfully: %s", file_name)
        return {"message": f"File '{file_name}' deleted successfully."}
    except Exception as e:
        logger.error("Error deleting file %s: %s", file_name, e)
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
@app.post("/askme/", summary="Query the Uploaded CSV File", tags=["Data Query"])
async def askme(
    question: str = Query(..., description="The question to ask about the CSV data"),
    file_name: str = Query("transactions.csv", description="Name of the CSV file to query")
):
    """
    Query the data in an uploaded CSV file using an LLM.
    
    **Parameters:**
    - **question**: The question related to the data in the csv file.
    - **file_name**: The CSV file to query (default: `transactions.csv`).
    
    The endpoint reads all sheets from the specified CSV file, then processes the question using the LLM function.
    """
    file_path = os.path.join('uploads', file_name)
    logger.info("Received query: '%s' for file: %s", question, file_path)
    
    if os.path.isfile(file_path):
        try:
            df = pd.read_csv(file_path)
            logger.info("CSV file read successfully: %s", file_name)
            # Process the question using the LLM function from utils
            result = LLM_GROQ(question, df,file_path)
            logger.info("LLM processed the query successfully.")
            return {"message": result}
        except Exception as e:
            logger.error("Error processing the CSV file %s: %s", file_name, e)
            raise HTTPException(status_code=500, detail=f"Error processing the CSV file: {str(e)}")
    else:
        logger.error("File not found: %s", file_path)
        raise HTTPException(status_code=404, detail="File not found. Please upload the file first.")

# Run the FastAPI app (if running locally)
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the FastAPI application on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

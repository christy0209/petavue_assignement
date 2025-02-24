from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import openai
import os
import random
import uuid
from datetime import datetime, timedelta
import time
import shutil
from utils import LLM

# Initialize FastAPI
app = FastAPI()

# Allowed content types for Excel files
ALLOWED_CONTENT_TYPES = {
    "application/vnd.ms-excel",  # .xls
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # .xlsx
}

@app.post("/upload/")
async def upload_excel(file: UploadFile = File(...)):
    # Verify the content type of the uploaded file
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel file.")

    # Alternatively, you could check the file extension if needed:
    filename = file.filename
    if not (filename.endswith(".xls") or filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="File extension is not allowed. Please upload an Excel file.")

    # Save the file to disk
    # Here, we create a temporary file, you can change the destination as needed.
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()

    return JSONResponse(content={"message": "File uploaded and saved successfully", "filename": filename})

@app.delete("/delete/")
async def delete_file(file_name: str="transactions.xlsx"):
    # Construct the full file path
    file_path = os.path.join('uploads', file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    try:
        os.remove(file_path)
        return {"message": f"File '{file_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
@app.post("/askme/")
async def askme(question: str = "",file_name: str = "transactions.xlsx"):
    file_path = os.path.join('uploads', file_name)
    # Check if the file exists and is a file
    if os.path.isfile(file_path):
        df = pd.read_excel(file_path, sheet_name=None)
        LLM(question,df)        
    else:
        raise HTTPException(status_code=404, detail="File not found. Please upload")



# Run the FastAPI app (if running locally)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

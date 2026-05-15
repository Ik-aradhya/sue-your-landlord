import asyncio
import os
import io
from fastapi import UploadFile
from backend.rag.ingest import run_ingestion_pipeline
from backend.models.schemas import DocType

async def main():
    laws_dir = "data/laws"
    if not os.path.exists(laws_dir):
        print(f"Directory {laws_dir} not found.")
        return

    for filename in os.listdir(laws_dir):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(laws_dir, filename)
        
        # Deduce state from filename
        state = "maharashtra"
        for s in ["maharashtra", "gujarat", "delhi", "karnataka", "tamil_nadu"]:
            if s in filename.lower().replace(" ", "_"):
                state = s
                break
        print(f"Ingesting {filename} for state: {state}...")
        
        with open(filepath, "rb") as f:
            content = f.read()
            
        # Mocking FastAPI's UploadFile
        from starlette.datastructures import Headers
        upload_file = UploadFile(
            filename=filename, 
            file=io.BytesIO(content),
            headers=Headers({"content-type": "application/pdf"})
        )
        
        result = await run_ingestion_pipeline(
            file=upload_file,
            doc_type=DocType.LAW,
            state=state,
            session_id=None
        )
        
        print(f"Result for {filename}: {result}")

if __name__ == "__main__":
    asyncio.run(main())

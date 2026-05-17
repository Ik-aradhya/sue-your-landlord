import asyncio
import os
import io
from fastapi import UploadFile
from backend.rag.ingest import run_ingestion_pipeline
from backend.models.schemas import DocType
from starlette.datastructures import Headers

async def main():
    laws_dir = "data/laws"
    if not os.path.exists(laws_dir):
        print(f"Directory {laws_dir} not found.")
        return

    for filename in os.listdir(laws_dir):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(laws_dir, filename)
        
        # ✅ All 10 states
        state = "unknown"
        for s in [
            "maharashtra", "gujarat", "delhi", "karnataka",
            "tamil_nadu", "telangana", "haryana",
            "uttar_pradesh", "west_bengal", "kerala"
        ]:
            if s in filename.lower().replace(" ", "_"):
                state = s
                break

        if state == "unknown":
            print(f"⚠️  Skipping {filename} — state not recognised")
            continue
            
        print(f"⏳ Ingesting {filename} → state: {state}")
        
        with open(filepath, "rb") as f:
            content = f.read()
            
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
        
        print(f"✅ {filename}: {result}")

if __name__ == "__main__":
    asyncio.run(main())
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router, status_router

app = FastAPI(
    title="Sue Your Landlord",
    description="AI-powered legal clarity engine for Indian tenants",
    version="1.0.0"
)

# CORS — allows your frontend to talk to this backend
# During development, allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1")
app.include_router(status_router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok", "product": "Sue Your Landlord v1"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

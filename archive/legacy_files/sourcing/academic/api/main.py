from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router as search_router

app = FastAPI(
    title="Academic Sourcing API",
    description="API for sourcing academic literature and datasets",
    version="1.0.0",
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:4321",
    "http://127.0.0.1:4321",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:4322",
    "http://127.0.0.1:4322",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(search_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "academic-sourcing"}

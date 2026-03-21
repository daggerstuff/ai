"""
Pixelated Empathy AI - Vercel Entry Point

Exports a Starlette ASGI app as the handler for @vercel/python.
FastAPI/Starlette are ASGI-native and work directly with Vercel's
serverless runtime.
"""

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


async def root(request):
    return JSONResponse({"status": "ok", "service": "pixelated-empathy-ai"})


async def health(request):
    return JSONResponse({"status": "healthy"})


routes = [
    Route("/", root),
    Route("/health", health),
]

middleware = [
    # NOTE: For production, restrict allow_origins to known frontend domains
    # Current wildcard is for development convenience
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware)

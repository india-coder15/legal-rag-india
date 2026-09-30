"""
Start the Legal AI REST API server.

Usage:
    python run_api.py

The API will be available at http://localhost:8080
Interactive docs at http://localhost:8080/docs
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.routes:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )

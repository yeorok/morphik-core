from fastapi import FastAPI, Depends
from .api import app as api_app
from .auth import DataBridgeAuth
import os

app = FastAPI()
auth = DataBridgeAuth(secret_key=os.getenv("JWT_SECRET_KEY", "your-secret-key"))

# Mount the API with authentication
app.mount("/api/v1", api_app)

# Add authentication middleware to all routes
@app.middleware("http")
async def authenticate_requests(request: Request, call_next):
    if request.url.path.startswith("/api/v1"):
        try:
            await auth(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
    return await call_next(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("ML_HOST", "127.0.0.1"), port=int(os.getenv("ML_PORT", "8000")), reload=os.getenv("ML_RELOAD", "false").lower() == "true")

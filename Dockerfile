# Dockerfile for CodeReview-Env OpenEnv Backend

FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying source for better caching
COPY pyproject.toml .

RUN pip install --no-cache-dir "pydantic>=2.0.0" "fastapi>=0.100.0" "uvicorn>=0.23.0" "thefuzz>=0.19.0"

# Copy the environment library
COPY . /app

# Expose the backend port
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

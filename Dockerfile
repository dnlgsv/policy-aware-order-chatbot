FROM python:3.12-slim

# set work directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv package manager
RUN pip install --no-cache-dir uv==0.4.29

# copy dependency files
COPY pyproject.toml uv.lock ./

# install Python dependencies
RUN uv sync --frozen --no-dev

# copy application code
COPY . .

# create non-root user for security
RUN useradd --create-home --shell /bin/bash chatbot && \
    chown -R chatbot:chatbot /app
USER chatbot

# expose port
EXPOSE 8000

# health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Start with slim Python 3.12 image
FROM python:3.12.12-slim

# Copy uv binary from official uv image (multi-stage build pattern)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Set working directory
WORKDIR /app

# Add virtual environment to PATH so we can use installed packages
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files first (better layer caching)
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies from lock file (ensures reproducible builds)
RUN uv sync --locked

# Copy directories and their content
COPY api/ api/
COPY models/ models/

# Set working directory
WORKDIR /app/api

# Set exposed port
EXPOSE 8000

# Set CMD to launch the server along with container launching
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
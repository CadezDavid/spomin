FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src ./src

# Install the project and dependencies into the system python
# This avoids venv PATH issues in Docker
RUN uv pip install --system .

# Create a non-root user
RUN groupadd -r spomin && useradd -r -g spomin spomin
RUN chown -R spomin:spomin /app

# Switch to non-root user
USER spomin

# Run the server
CMD ["spomin"]
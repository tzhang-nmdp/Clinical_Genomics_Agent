FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: node (for npx/github MCP), uv, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && pip install uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (CPU-only FAISS; no heavy CUDA torch needed for inference)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# # Install geocode-mcp from local source
# COPY geocode-mcp/ geocode-mcp/
# RUN pip install --no-cache-dir ./geocode-mcp

# Copy application code
COPY agent.md agent.md
COPY api_key.json api_key.json
COPY mcp_servers_docker.json mcp_servers.json
COPY mcp_client.py mcp_client.py
COPY server.py server.py
COPY docker-compose.yml docker-compose.yml
COPY activity_logger.py activity_logger.py

COPY subagents/ subagents/
COPY tools/ tools/
COPY skills/ skills/
COPY static/ static/
COPY mcp_servers/ mcp_servers/
RUN mkdir logs/
EXPOSE 8000

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

#!/bin/bash
set -e

echo "Waiting for llama.cpp server to be ready..."
until curl -sf http://llama:8080/health > /dev/null; do
  sleep 2
done

until curl -sf http://llama2:8081/health > /dev/null; do
  sleep 2
done

echo "llama.cpp server is ready. Starting FastAPI server..."
exec uvicorn server:app --host 0.0.0.0 --port 8000

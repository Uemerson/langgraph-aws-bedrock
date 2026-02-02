docker build -t langgraph-aws-bedrock -f Dockerfile.dev .
docker run --rm \
    --env-file .env.dev \
    -p 8000:8000 \
    -v "$(pwd):/app" \
    langgraph-aws-bedrock
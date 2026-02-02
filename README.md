# LangGraph + AWS Bedrock Conversation API

A production-ready FastAPI service that orchestrates **multi-step conversation workflows** using **LangGraph** and **AWS Bedrock**. The service implements intelligent agent workflows with context validation, hybrid vector search retrieval, and streaming responses via Server-Sent Events (SSE).

## Overview

This project demonstrates a complete AI conversation system with:
- **LangGraph Workflows**: Multi-step agent execution with conditional routing and state management
- **AWS Bedrock Integration**: Powered by Claude/Nova models for intelligent reasoning
- **Hybrid Vector Search**: Dual-index retrieval combining dense and sparse embeddings via Pinecone
- **SSE Streaming**: Real-time responses streamed to clients
- **LangSmith Observability**: Full tracing and usage analytics
- **Knowledge Base Management**: Document upload and RAG retrieval system
- **Modular Architecture**: Clean separation of concerns with services, routes, and workflows

## Features

### Core Capabilities
- **FastAPI Framework**: High-performance async REST API
- **Multi-Node LangGraph Workflow**: Intelligent 4-node agent with conditional routing
  - Context validation node: Checks if input has sufficient context
  - RAG retrieval node: Hybrid search with deduplication and reranking
  - Answer generation node: Streaming response with context awareness
  - Fallback node: Graceful handling when context is insufficient
- **Conditional Edge Routing**: Dynamic path selection based on state conditions
- **AWS Bedrock LLM**: Integration with `us.amazon.nova-2-lite-v1:0` model
- **Pinecone Hybrid Search**: Dual-index setup for optimal relevance
  - Dense embeddings via Llama Text Embed V2
  - Sparse embeddings via Pinecone Sparse English V0
  - BGE Reranker V2-M3 for result optimization
- **Document Processing**: Extract and chunk text from PDF, DOCX, and TXT files
- **Token Tracking**: LangSmith integration for usage analytics
- **CORS Enabled**: Cross-origin support for web clients

## Architecture

### Workflow Data Flow
```
User Request (/conversation)
    ▼
ConversationService.converse()
    ▼
AgentWorkFlow.stream() (LangGraph)
    ├──▶ check_context_node: Validates if prompt has sufficient context
    │                       (Uses AWS Bedrock to check context relevance)
    │
    ├──▶ retrieve_rag_node: Hybrid search with deduplication & reranking
    │                       - Dense embeddings search
    │                       - Sparse embeddings search
    │                       - BGE Reranker V2-M3 for top-k optimization
    │
    ├──▶ generate_answer_node: Streams response with RAG context
    │                          - ChatPromptTemplate with system instructions
    │                          - Async streaming from AWS Bedrock
    │                          - Token counting & LangSmith tracking
    │
    ├──▶ cannot_answer_node: Fallback response if context/docs insufficient
    ▼
StreamingResponse (SSE)
    ▼
Client (real-time text chunks)
```

### Component Breakdown

| Component | Purpose |
|-----------|---------|
| `src/main.py` | FastAPI app initialization with route registration |
| `src/dependencies/app.py` | Settings, AWS Bedrock client, Pinecone setup, lifespan management |
| `src/routes/conversation.py` | `POST /conversation` endpoint (streaming) |
| `src/routes/knowledge_base.py` | `POST /knowledgebase/upload` endpoint (file upload) |
| `src/services/conversation_service.py` | Orchestrates workflow streaming |
| `src/services/knowledge_base_service.py` | Document extraction, chunking, and Pinecone ingestion |
| `src/workflows/agent_workflow.py` | 4-node LangGraph state machine with conditional routing |

## Installation

### Prerequisites
- Python 3.12+
- AWS Bedrock access (with `nova-2-lite-v1` model enabled)
- Pinecone account with API key
- (Optional) LangSmith credentials for observability

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd langgraph-aws-bedrock
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Configure environment variables (`.env`):
```env
# AWS Bedrock Configuration
AWS_BEARER_TOKEN_BEDROCK=your-bearer-token
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key

# LangSmith Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=your-project-name

# API Configuration
BACKEND_CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
```

## Running Locally

### Development Server
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at `http://localhost:8000`

### Test Endpoints

**Conversation Streaming:**
```bash
curl -N -H "Content-Type: application/json" \
  -d '{"message":"What is artificial intelligence?"}' \
  http://localhost:8000/conversation
```

**Upload Knowledge Base Document:**
```bash
curl -X POST -F "file=@document.pdf" \
  http://localhost:8000/knowledgebase/upload
```

Supported formats: PDF, DOCX, TXT

## Docker Deployment

### Build Image
```bash
docker build -t langgraph-aws-bedrock -f Dockerfile.dev .
```

### Run Container
```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  -v "$(pwd)/src:/app/src" \
  langgraph-aws-bedrock
```

Or use the provided script:
```bash
bash ./up.sh
```

## API Documentation

### `/conversation` - POST
Initiate streaming conversation with the agent.

**Request:**
```json
{
  "message": "Your question or prompt"
}
```

**Response:** `text/event-stream`
- Headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`
- Body: Server-Sent Events format
  - Each chunk: `data: <response-text>\n\n`

**Example (JavaScript/TypeScript):**
```typescript
const response = await fetch('http://localhost:8000/conversation', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Explain quantum computing' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const text = line.replace('data: ', '');
      console.log(text);
    }
  }
}
```

### `/knowledgebase/upload` - POST
Upload documents to the knowledge base for RAG retrieval.

**Request:**
- Form-data: `file` (PDF, DOCX, or TXT)

**Response:**
```json
{
  "status": "success"
}
```

**Example (cURL):**
```bash
curl -X POST -F "file=@research_paper.pdf" \
  http://localhost:8000/knowledgebase/upload
```

## Project Structure

```
.
├── src/
│   ├── main.py                      # FastAPI app entry point
│   ├── dependencies/
│   │   └── app.py                   # Settings, clients, middleware
│   ├── routes/
│   │   ├── conversation.py          # /conversation endpoint
│   │   └── knowledge_base.py        # /knowledgebase/upload endpoint
│   ├── services/
│   │   ├── conversation_service.py  # Streaming orchestration
│   │   └── knowledge_base_service.py # Document processing & ingestion
│   └── workflows/
│       └── agent_workflow.py        # 4-node LangGraph state machine
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (not committed)
├── Dockerfile.dev                  # Development container
├── up.sh                           # Docker run script
├── docker.clean.sh                 # Docker cleanup script
└── README.md                       # This file
```

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI 0.128.0 |
| **Agent Orchestration** | LangGraph 1.0.7 |
| **LLM** | AWS Bedrock (Nova 2 Lite) |
| **Vector Database** | Pinecone 8.0.0 |
| **Text Processing** | LangChain 1.2.7 |
| **Document Parsing** | PyPDF, python-docx |
| **Observability** | LangSmith 0.6.6 |
| **Web Server** | Uvicorn 0.40.0 |
| **Containerization** | Docker |

## Key Features Explained

### Hybrid Vector Search
- **Dense Index**: Semantic similarity using Llama Text Embed V2
- **Sparse Index**: Keyword matching using Pinecone Sparse English V0
- **Reranking**: BGE Reranker V2-M3 ranks top results for relevance

### Document Processing
Supports three file formats:
- **PDF**: Text extraction via PyPDF
- **DOCX**: Paragraph extraction via python-docx
- **TXT**: Direct UTF-8 decoding

All documents are:
1. Split into 500-char chunks with 50-char overlap
2. Stored with unique IDs and source filename
3. Indexed in both dense and sparse Pinecone indexes

### Streaming Response
- Real-time token-by-token delivery via SSE
- Async streaming from AWS Bedrock
- Token counting and LangSmith metadata tracking
- Automatic error handling with fallback messages

### Context Validation
The `check_context_node` uses AWS Bedrock to determine if the user input has sufficient context. Routing:
- **YES** → Proceed to RAG retrieval
- **NO** → Return "cannot answer" message

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_BEARER_TOKEN_BEDROCK` | Yes | AWS Bedrock authentication token |
| `AWS_ACCESS_KEY_ID` | Yes | AWS access key for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS secret key for Bedrock |
| `PINECONE_API_KEY` | Yes | Pinecone API key for vector store |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (default: true) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `BACKEND_CORS_ORIGINS` | No | CORS allowed origins (default: `["*"]`) |

## Troubleshooting

### Issue: AWS Bedrock Authentication Error
- Verify AWS credentials in `.env`
- Check IAM permissions for Bedrock model access
- Ensure region is `us-east-1`

### Issue: Pinecone Connection Failed
- Confirm `PINECONE_API_KEY` is correct
- Check Pinecone cloud region settings
- Verify indexes exist or will be created on startup

### Issue: Document Upload Fails
- Ensure file format is PDF, DOCX, or TXT
- Check file is not corrupted
- Verify Pinecone indexes are initialized

### Issue: Slow Responses
- Check Pinecone index sizes (high latency if large)
- Review LangSmith traces for bottlenecks
- Consider adjusting chunk size/overlap in `knowledge_base_service.py`

## Development Notes

### Adding New Nodes
To add a new node to the workflow:

1. Define node method in `AgentWorkFlow`:
```python
def custom_node(self, state: GraphState) -> Dict:
    # Process state
    return {"key": value}
```

2. Register in `__build_graph()`:
```python
workflow.add_node("custom", self.custom_node)
```

3. Add edges to connect the node

### Customizing Prompts
Edit prompts in the respective nodes:
- **Context check**: `check_context_node`
- **Answer generation**: `generate_answer_node`

### Monitoring with LangSmith
Enable tracing to see:
- Full execution traces
- Token usage per operation
- Latency metrics
- Error logs

## Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Support
For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review LangSmith traces for detailed execution info
- Consult AWS Bedrock documentation
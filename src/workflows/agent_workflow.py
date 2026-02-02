"""Workflow Agent using LangGraph and Google GenAI."""

from typing import Dict, List, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import get_current_run_tree
from langsmith.schemas import UsageMetadata
from pinecone import Pinecone


class GraphState(TypedDict):
    """Graph State."""

    prompt: str
    answer: str
    has_context: bool
    has_documents: bool
    documents: List[Dict]


class AgentWorkFlow:
    """Agent Workflow to manage a LangGraph execution."""

    __client: ChatBedrockConverse
    __model_id: str
    __pinecone: Pinecone
    __app: CompiledStateGraph

    def __init__(
        self, client: ChatBedrockConverse, model_id: str, pinecone: Pinecone
    ):
        self.__client = client
        self.__model_id = model_id
        self.__pinecone = pinecone
        self.__app = self.__build_graph()

    def __build_graph(self) -> CompiledStateGraph:
        """Graph construction."""

        workflow = StateGraph(GraphState)

        workflow.add_node("check_context", self.check_context_node)
        workflow.add_node("retrieve_rag", self.retrieve_rag_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("cannot_answer", self.cannot_answer_node)

        workflow.add_conditional_edges(
            "check_context",
            self.check_context_condition,
            {
                "has_context": "retrieve_rag",
                "no_context": "cannot_answer",
            },
        )

        workflow.add_conditional_edges(
            "retrieve_rag",
            self.retrieve_rag_condition,
            {
                "has_documents": "generate_answer",
                "no_documents": "cannot_answer",
            },
        )

        workflow.add_edge("generate_answer", END)
        workflow.set_entry_point("check_context")

        return workflow.compile()

    def check_context_node(self, state: GraphState) -> Dict:
        """Checks whether the prompt has enough context."""
        response = self.__client.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Does the following input "
                        "contain a clear question or topic with enough "
                        "context to answer? "
                        "Answer ONLY with exactly one token:"
                        "YES or NO."
                        "No punctuation, no explanation, no extra text."
                        "If unsure, answer NO."
                    ),
                },
                {"role": "user", "content": state["prompt"]},
            ],
        )

        input_tokens = response.usage_metadata["input_tokens"]
        output_tokens = response.usage_metadata["output_tokens"]

        self.__usage_metadata(input_tokens, output_tokens)

        return {"has_context": "YES" in response.content[0]["text"].upper()}

    def __merge_chunks(self, h1: Dict, h2: Dict) -> List[Dict]:
        """
        Get the unique hits from two search results and
        return them as single array of {'_id', 'chunk_text'} dicts,
        printing each dict on a new line.
        """

        # Deduplicate by _id
        deduped_hits = {
            hit["_id"]: hit
            for hit in h1["result"]["hits"] + h2["result"]["hits"]
        }.values()

        # Sort by _score descending
        sorted_hits = sorted(
            deduped_hits, key=lambda x: x["_score"], reverse=True
        )

        # Transform to format for reranking
        result = [
            {"_id": hit["_id"], "chunk_text": hit["fields"]["chunk_text"]}
            for hit in sorted_hits
        ]
        return result

    def retrieve_rag_node(self, state: GraphState) -> Dict:
        """RAG retrieval."""

        query = state["prompt"].lower()

        dense_index = self.__pinecone.Index(
            "dense-for-hybrid-knowledge-base-index"
        )
        sparse_index = self.__pinecone.Index(
            "sparse-for-hybrid-knowledge-base-index"
        )
        namespace = "rag-namespace"
        dense_results = dense_index.search(
            namespace=namespace,
            query={"top_k": 40, "inputs": {"text": query}},
        )
        sparse_results = sparse_index.search(
            namespace=namespace,
            query={"top_k": 40, "inputs": {"text": query}},
        )

        if (
            len(dense_results["result"]["hits"]) == 0
            and len(sparse_results["result"]["hits"]) == 0
        ):
            return {"has_documents": False}

        merged_results = self.__merge_chunks(sparse_results, dense_results)

        result = self.__pinecone.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=merged_results,
            rank_fields=["chunk_text"],
            top_n=10,
            return_documents=True,
            parameters={"truncate": "END"},
        )

        documents = result.data

        return {"has_documents": bool(documents), "documents": documents}

    async def generate_answer_node(self, state: GraphState) -> Dict:
        """Generates the final answer."""

        writer = get_stream_writer()
        response_text = ""

        input_tokens = 0
        output_tokens = 0

        context_text = "\n".join(
            [doc["document"]["chunk_text"] for doc in state["documents"]]
        )

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a strict virtual assistant. "
                        "Use ONLY the content contained within "
                        "the <context> tags to answer.\n"
                        "If the answer cannot be found in the context, "
                        "respond exactly: 'I’m sorry, but I don’t have enough "
                        "information in the documents to answer that.'\n"
                        "Do not use any knowledge beyond "
                        "what has been provided."
                    ),
                ),
                (
                    "human",
                    (
                        "Here is the context for reference:\n"
                        "<context>\n"
                        "{context}\n"
                        "</context>\n\n"
                        "Based on the context above, answer: {question}"
                    ),
                ),
            ]
        )

        inputs = {
            "context": context_text,
            "question": state["prompt"],
        }
        messages = rag_prompt.format_messages(**inputs)

        async for chunk in self.__client.astream(messages):
            if chunk.content:
                response_text += chunk.content[0]["text"]
                writer({"answer": chunk.content[0]["text"]})

            if chunk.usage_metadata:
                input_tokens += chunk.usage_metadata["input_tokens"]
                output_tokens += chunk.usage_metadata["output_tokens"]

        self.__usage_metadata(input_tokens, output_tokens)

        return {"answer": response_text}

    def __usage_metadata(self, input_tokens: int, output_tokens: int) -> None:
        """Send usage metadata to LangSmith."""

        run = get_current_run_tree()

        if run:
            run.add_metadata(
                metadata={
                    "ls_model_name": self.__model_id,
                    "ls_model_type": "llm",
                    "ls_provider": "google_genai",
                    "ls_run_depth": 0,
                    "ls_temperature": 0.7,
                    "usage_metadata": UsageMetadata(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                    "invocation_params": {
                        "_type": "google_gemini",
                        "candidate_count": 1,
                        "image_config": None,
                        "max_output_tokens": None,
                        "model": self.__model_id,
                        "stop": None,
                        "temperature": 0.7,
                        "top_k": None,
                        "top_p": None,
                    },
                    "options": {"streaming": True, "stop": None},
                },
            )

    def cannot_answer_node(self, _: GraphState) -> Dict:
        """Fallback node when the question cannot be answered."""

        return {
            "answer": (
                "I'm sorry, but I cannot provide an answer "
                "based on the given input."
            )
        }

    def check_context_condition(self, state: GraphState) -> str:
        """Determines the condition based on context check."""

        return "has_context" if state["has_context"] else "no_context"

    def retrieve_rag_condition(self, state: GraphState) -> str:
        """Determines the condition based on RAG retrieval."""

        return "has_documents" if state["has_documents"] else "no_documents"

    async def stream(self, prompt: str):
        """Streams workflow execution output."""

        async for stream_mode, chunk in self.__app.astream(
            input={"prompt": prompt},
            stream_mode=[
                "values",
                "updates",
                "custom",
                "messages",
                "checkpoints",
                "tasks",
            ],
        ):
            if stream_mode in ["custom", "values"]:
                if chunk.get("answer"):
                    yield chunk["answer"]

    def save_graph(self, path: str = "graph.png") -> None:
        """Saves the workflow graph as a Mermaid PNG."""

        png = self.__app.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png)

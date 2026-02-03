# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom RAG implementation by Frank Reply for Tokkio ACE Controller with:
# - Redis vector store support
# - Citation support for Tokkio UI
# - Multi-turn conversation handling

import logging
import os
from typing import Any, Dict, Generator, List

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Redis vector store - try community first, fall back to langchain
try:
    from langchain_community.vectorstores.redis import Redis as RedisVectorStore
except ImportError:
    from langchain.vectorstores.redis import Redis as RedisVectorStore

from RAG.src.chain_server.base import BaseExample
from RAG.src.chain_server.tracing import langchain_instrumentation_class_wrapper
from RAG.src.chain_server.utils import (
    get_config,
    get_embedding_model,
    get_llm as nvidia_get_llm,
    get_prompts,
)


def get_llm(**kwargs):
    """Get LLM instance, supporting both NVIDIA and OpenAI."""
    settings = get_config()
    model_engine = getattr(settings.llm, 'model_engine', 'nvidia-ai-endpoints')
    model_name = getattr(settings.llm, 'model_name', 'meta/llama-3.1-8b-instruct')
    
    logger.info(f"Getting LLM with engine={model_engine}, model={model_name}")
    
    if model_engine == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=0.7,
                streaming=True,
            )
        except ImportError:
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                streaming=True,
            )
    else:
        # Fall back to NVIDIA's get_llm for nvidia-ai-endpoints
        return nvidia_get_llm(**kwargs)

logger = logging.getLogger(__name__)
settings = get_config()
prompts = get_prompts()
document_embedder = get_embedding_model()

# Redis connection settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "tokkio_docs")

# Initialize vector store
vectorstore = None

try:
    import redis
    
    logger.info(f"Connecting to Redis at {REDIS_URL.split('@')[1] if '@' in REDIS_URL else REDIS_URL}")
    
    # Test basic connectivity
    test_client = redis.from_url(REDIS_URL)
    test_client.ping()
    logger.info("Redis connection test successful")
    
    # Create the vector store
    vectorstore = RedisVectorStore(
        redis_url=REDIS_URL,
        index_name=REDIS_INDEX_NAME,
        embedding=document_embedder,
    )
    logger.info(f"Connected to Redis vector store with index '{REDIS_INDEX_NAME}'")
except Exception as e:
    logger.warning(f"Unable to connect to Redis during initialization: {e}")
    logger.warning("Vector store will not be available. Documents cannot be ingested or searched.")


@langchain_instrumentation_class_wrapper
class TokkioRAG(BaseExample):
    """
    Tokkio-compatible RAG implementation with:
    - Redis as vector store
    - Citation support (returns source documents with scores)
    - Multi-turn conversation context
    """

    def ingest_docs(self, filepath: str, filename: str) -> None:
        """Ingest documents into Redis vector store."""
        
        if not filename.endswith((".txt", ".pdf", ".md", ".html")):
            raise ValueError(f"{filename} is not a supported file type")
        
        try:
            # Load document
            raw_documents = UnstructuredFileLoader(filepath).load()
            
            if not raw_documents:
                logger.warning("No documents found to process")
                return
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            documents = text_splitter.split_documents(raw_documents)
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = filename
                doc.metadata["filename"] = filename
            
            # Add to Redis
            global vectorstore
            if vectorstore is None:
                # Create new index
                vectorstore = RedisVectorStore.from_documents(
                    documents,
                    document_embedder,
                    redis_url=REDIS_URL,
                    index_name=REDIS_INDEX_NAME,
                )
            else:
                vectorstore.add_documents(documents)
            
            logger.info(f"Ingested {len(documents)} chunks from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            raise ValueError(f"Failed to upload document: {e}")

    def llm_chain(
        self, query: str, chat_history: List["Message"], **kwargs
    ) -> Generator[str, None, None]:
        """Execute LLM chain without knowledge base."""
        
        logger.info("Using LLM without knowledge base")
        
        system_message = [("system", prompts.get("chat_template", "You are a helpful assistant."))]
        
        # Include conversation history
        conversation_history = [(msg.role, msg.content) for msg in chat_history[-10:]]  # Last 10 messages
        user_input = [("user", "{input}")]
        
        prompt_template = ChatPromptTemplate.from_messages(
            system_message + conversation_history + user_input
        )
        
        llm = get_llm(**kwargs)
        chain = prompt_template | llm | StrOutputParser()
        
        return chain.stream({"input": query}, config={"callbacks": [self.cb_handler]})

    def rag_chain(
        self, query: str, chat_history: List["Message"], **kwargs
    ) -> Generator[str, None, None]:
        """
        Execute RAG chain with citation support for Tokkio.
        
        This method retrieves relevant documents, includes them in the prompt,
        and yields the response. Citations are stored for later retrieval.
        """
        
        logger.info("Using RAG to generate response")
        
        if vectorstore is None:
            logger.warning("Vector store not initialized")
            return iter(["I don't have access to a knowledge base. Please upload some documents first."])
        
        try:
            # Retrieve relevant documents
            top_k = getattr(settings.retriever, 'top_k', 4)
            score_threshold = getattr(settings.retriever, 'score_threshold', 0.3)
            
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": score_threshold,
                    "k": top_k,
                },
            )
            
            docs = retriever.invoke(query)
            
            if not docs:
                logger.warning("No relevant documents found")
                return iter(["I couldn't find relevant information in my knowledge base for your question."])
            
            # Build context from retrieved documents
            context_parts = []
            self._last_citations = []  # Store for citation retrieval
            
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", doc.metadata.get("filename", f"doc_{i}"))
                content = doc.page_content
                score = doc.metadata.get("score", 0.0)
                
                context_parts.append(f"[Source: {source}]\n{content}")
                
                # Store citation for Tokkio
                self._last_citations.append({
                    "document_type": "text",
                    "document_id": str(i),
                    "document_name": source,
                    "content": content,
                    "metadata": str(doc.metadata),
                    "score": score,
                })
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Build prompt with context and conversation history
            rag_template = prompts.get("rag_template", """Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {input}

Answer:""")
            
            # Include recent conversation history
            conversation_history = [(msg.role, msg.content) for msg in chat_history[-6:]]
            
            system_message = [("system", rag_template)]
            history_messages = conversation_history if conversation_history else []
            user_message = [("user", "{input}")]
            
            prompt_template = ChatPromptTemplate.from_messages(
                system_message + history_messages + user_message
            )
            
            llm = get_llm(**kwargs)
            chain = prompt_template | llm | StrOutputParser()
            
            logger.debug(f"RAG context: {context[:500]}...")
            logger.debug(f"RAG query: {query}")
            
            # Pass both context and input to the template
            return chain.stream({"context": context, "input": query}, config={"callbacks": [self.cb_handler]})
            
        except Exception as e:
            logger.error(f"RAG chain error: {e}")
            return iter([f"I encountered an error while searching: {str(e)}"])

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        
        if vectorstore is None:
            return []
        
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs})
            docs = retriever.invoke(content)
            
            results = []
            for doc in docs:
                results.append({
                    "source": doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
                    "content": doc.page_content,
                    "score": doc.metadata.get("score", 0.0),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return []

    def get_documents(self) -> List[str]:
        """Get list of ingested documents."""
        
        if vectorstore is None:
            return []
        
        try:
            # Redis doesn't have a direct way to list all documents
            # This is a limitation - we'd need to maintain a separate index
            # For now, return empty list
            logger.warning("get_documents not fully implemented for Redis")
            return []
        except Exception as e:
            logger.error(f"Get documents error: {e}")
            return []

    def delete_documents(self, filenames: List[str]) -> bool:
        """Delete documents from vector store."""
        
        if vectorstore is None:
            return False
        
        try:
            # Redis deletion by metadata requires custom implementation
            logger.warning("delete_documents not fully implemented for Redis")
            return False
        except Exception as e:
            logger.error(f"Delete documents error: {e}")
            return False

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom RAG implementation by Frank Reply for Tokkio ACE Controller with:
# - Oracle 26ai vector store (semantic or hybrid search)
# - Citation support for Tokkio UI
# - Multi-turn conversation handling

import array
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
                request_timeout=30,
                max_retries=2,
            )
        except ImportError:
            raise ImportError("OpenAI engine requires langchain-openai. Install with: pip install langchain-openai")
    else:
        return nvidia_get_llm(**kwargs)


logger = logging.getLogger(__name__)
settings = get_config()

prompts = get_prompts()
document_embedder = get_embedding_model()

# ---------------------------------------------------------------------------
# Oracle 26ai vector store
# ---------------------------------------------------------------------------

ORACLE_DB_USER = os.getenv("ORACLE_DB_USER", "ADMIN")
ORACLE_DB_PASSWORD = os.getenv("ORACLE_DB_PASSWORD", "")
ORACLE_DB_DSN = os.getenv("ORACLE_DB_DSN", "")
ORACLE_WALLET_DIR = os.getenv("ORACLE_WALLET_DIR", "")
ORACLE_WALLET_PASSWORD = os.getenv("ORACLE_WALLET_PASSWORD", "")
ORACLE_TABLE_NAME = os.getenv("ORACLE_TABLE_NAME", "RAG_CHUNKS")

# Hybrid search settings
ORACLE_SEARCH_MODE = os.getenv("ORACLE_SEARCH_MODE", "semantic")  # "semantic" or "hybrid"
ORACLE_DOCS_TABLE = os.getenv("ORACLE_DOCS_TABLE", "RAG_DOCS")
ORACLE_ONNX_MODEL = os.getenv("ORACLE_ONNX_MODEL", "MULTILINGUAL_E5_BASE")
ORACLE_HYBRID_INDEX = f"{ORACLE_DOCS_TABLE}_HIDX"

_oracle_pool = None

def _get_oracle_pool():
    """Lazy-init a connection pool to Oracle 26ai."""
    global _oracle_pool
    if _oracle_pool is not None:
        return _oracle_pool

    import oracledb
    oracledb.defaults.fetch_lobs = False

    connect_kwargs = dict(
        user=ORACLE_DB_USER,
        password=ORACLE_DB_PASSWORD,
        dsn=ORACLE_DB_DSN,
        min=1,
        max=4,
        increment=1,
    )
    if ORACLE_WALLET_DIR:
        connect_kwargs.update(
            config_dir=ORACLE_WALLET_DIR,
            wallet_location=ORACLE_WALLET_DIR,
            wallet_password=ORACLE_WALLET_PASSWORD,
        )

    _oracle_pool = oracledb.create_pool(**connect_kwargs)
    logger.info(f"Oracle 26ai connection pool created (DSN ending ...{ORACLE_DB_DSN[-40:]})")
    return _oracle_pool


def _ensure_table():
    """Create the RAG_CHUNKS table and vector index if they don't exist."""
    pool = _get_oracle_pool()
    with pool.acquire() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = :1",
            [ORACLE_TABLE_NAME],
        )
        if cur.fetchone()[0] == 0:
            cur.execute(f"""
                CREATE TABLE {ORACLE_TABLE_NAME} (
                    chunk_id    VARCHAR2(200) PRIMARY KEY,
                    source      VARCHAR2(500),
                    filename    VARCHAR2(500),
                    content     CLOB,
                    embedding   VECTOR(3072, FLOAT32)
                )
            """)
            cur.execute(f"""
                CREATE VECTOR INDEX {ORACLE_TABLE_NAME}_vidx
                ON {ORACLE_TABLE_NAME} (embedding)
                ORGANIZATION NEIGHBOR PARTITIONS
                DISTANCE COSINE
                WITH TARGET ACCURACY 95
            """)
            conn.commit()
            logger.info(f"Created table {ORACLE_TABLE_NAME} with vector index")
        else:
            logger.info(f"Table {ORACLE_TABLE_NAME} already exists")


def _oracle_insert_chunks(chunks: List[Tuple[str, str, str, str, list]]):
    """Insert (chunk_id, source, filename, content, embedding_list) rows."""
    pool = _get_oracle_pool()
    with pool.acquire() as conn:
        cur = conn.cursor()
        for chunk_id, source, filename, content, emb in chunks:
            emb_arr = array.array("f", emb)
            cur.execute(
                f"MERGE INTO {ORACLE_TABLE_NAME} t "
                f"USING (SELECT :cid AS chunk_id FROM dual) s "
                f"ON (t.chunk_id = s.chunk_id) "
                f"WHEN NOT MATCHED THEN INSERT (chunk_id, source, filename, content, embedding) "
                f"VALUES (:cid, :src, :fn, :cnt, :emb)",
                {"cid": chunk_id, "src": source, "fn": filename, "cnt": content, "emb": emb_arr},
            )
        conn.commit()
    logger.info(f"Inserted/merged {len(chunks)} chunks into {ORACLE_TABLE_NAME}")


def _oracle_search(query_embedding: list, top_k: int = 4) -> List[Dict[str, Any]]:
    """Semantic similarity search. Returns list of dicts with content, source, score."""
    pool = _get_oracle_pool()
    q_arr = array.array("f", query_embedding)
    with pool.acquire() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT chunk_id, source, filename, content,
                   VECTOR_DISTANCE(embedding, :1, COSINE) AS dist
            FROM {ORACLE_TABLE_NAME}
            ORDER BY dist
            FETCH FIRST :2 ROWS ONLY
        """, [q_arr, top_k])
        results = []
        for row in cur:
            results.append({
                "chunk_id": row[0],
                "source": row[1],
                "filename": row[2],
                "content": row[3],
                "score": round(1.0 - row[4], 4),
            })
    return results


# ---------------------------------------------------------------------------
# Hybrid search (BM25 + vector via HYBRID VECTOR INDEX with ONNX model)
# ---------------------------------------------------------------------------

def _ensure_hybrid_table():
    """Create the RAG_DOCS table for hybrid search (full docs, no pre-chunking)."""
    pool = _get_oracle_pool()
    with pool.acquire() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = :1",
            [ORACLE_DOCS_TABLE],
        )
        if cur.fetchone()[0] == 0:
            cur.execute(f"""
                CREATE TABLE {ORACLE_DOCS_TABLE} (
                    doc_id    VARCHAR2(200) PRIMARY KEY,
                    source    VARCHAR2(500),
                    filename  VARCHAR2(500),
                    content   CLOB
                )
            """)
            conn.commit()
            logger.info(f"Created table {ORACLE_DOCS_TABLE}")

        cur.execute(
            "SELECT COUNT(*) FROM ctx_user_indexes WHERE idx_name = :1",
            [ORACLE_HYBRID_INDEX],
        )
        if cur.fetchone()[0] == 0:
            logger.info(f"Creating hybrid vector index {ORACLE_HYBRID_INDEX}...")
            cur.execute(f"""
                CREATE HYBRID VECTOR INDEX {ORACLE_HYBRID_INDEX}
                ON {ORACLE_DOCS_TABLE}(content)
                PARAMETERS('MODEL {ORACLE_ONNX_MODEL}')
            """)
            conn.commit()
            logger.info(f"Created hybrid vector index {ORACLE_HYBRID_INDEX}")
        else:
            logger.info(f"Hybrid index {ORACLE_HYBRID_INDEX} already exists")


def _hybrid_insert_docs(docs: List[Tuple[str, str, str, str]]):
    """Insert full documents (doc_id, source, filename, content) for hybrid indexing."""
    pool = _get_oracle_pool()
    with pool.acquire() as conn:
        cur = conn.cursor()
        for doc_id, source, filename, content in docs:
            cur.execute(
                f"MERGE INTO {ORACLE_DOCS_TABLE} t "
                f"USING (SELECT :did AS doc_id FROM dual) s "
                f"ON (t.doc_id = s.doc_id) "
                f"WHEN NOT MATCHED THEN INSERT (doc_id, source, filename, content) "
                f"VALUES (:did, :src, :fn, :cnt) "
                f"WHEN MATCHED THEN UPDATE SET content = :cnt",
                {"did": doc_id, "src": source, "fn": filename, "cnt": content},
            )
        conn.commit()
    logger.info(f"Inserted/merged {len(docs)} documents into {ORACLE_DOCS_TABLE}")


def _oracle_hybrid_search(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Hybrid search combining BM25 keyword matching and ONNX vector similarity.
    Uses Reciprocal Rank Fusion (RRF) to combine scores.
    """
    pool = _get_oracle_pool()
    search_params = json.dumps({
        "hybrid_index_name": ORACLE_HYBRID_INDEX,
        "search_text": query,
        "search_fusion": "UNION",
        "search_scorer": "RRF",
        "return": {
            "topN": top_k,
            "values": ["rowid", "score", "chunk_text", "chunk_id"],
            "format": "JSON",
        },
    })

    with pool.acquire() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:1)) FROM dual",
            [search_params],
        )
        result_clob = cur.fetchone()[0]
        if not result_clob:
            return []

        result_str = result_clob if isinstance(result_clob, str) else result_clob.read()
        search_results = json.loads(result_str)

        results = []
        for item in search_results:
            rowid = item.get("rowid")
            score = item.get("score", 0.0)
            chunk_text = item.get("chunk_text", "")

            source = ""
            filename = ""
            if rowid:
                cur.execute(
                    f"SELECT source, filename FROM {ORACLE_DOCS_TABLE} WHERE ROWID = :1",
                    [rowid],
                )
                row = cur.fetchone()
                if row:
                    source, filename = row[0] or "", row[1] or ""

            results.append({
                "chunk_id": item.get("chunk_id", ""),
                "source": source,
                "filename": filename,
                "content": chunk_text,
                "score": round(float(score), 4),
            })

    return results


# --- Initialize on import ---
_vectorstore_ready = False
try:
    if ORACLE_DB_DSN:
        if ORACLE_SEARCH_MODE == "hybrid":
            logger.info("Initializing Oracle 26ai in HYBRID search mode")
            _ensure_hybrid_table()
        else:
            logger.info("Initializing Oracle 26ai in SEMANTIC search mode")
            _ensure_table()
        _vectorstore_ready = True
    else:
        logger.warning("ORACLE_DB_DSN not set; vector store disabled")
except Exception as e:
    logger.warning(f"Unable to connect to Oracle 26ai during init: {e}")
    logger.warning("Vector store will not be available.")


@langchain_instrumentation_class_wrapper
class TokkioRAG(BaseExample):
    """
    Tokkio-compatible RAG implementation with:
    - Oracle 26ai as vector store (semantic search)
    - Citation support (returns source documents with scores)
    - Multi-turn conversation context
    """

    def ingest_docs(self, filepath: str, filename: str) -> None:
        """Ingest documents into Oracle 26ai vector store."""

        if not filename.endswith((".txt", ".pdf", ".md", ".html")):
            raise ValueError(f"{filename} is not a supported file type")

        try:
            raw_documents = UnstructuredFileLoader(filepath).load()
            if not raw_documents:
                logger.warning("No documents found to process")
                return

            if ORACLE_SEARCH_MODE == "hybrid":
                full_text = "\n\n".join(doc.page_content for doc in raw_documents)
                _hybrid_insert_docs([(filename, filename, filename, full_text)])
                logger.info(f"Ingested full document {filename} for hybrid search")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                documents = text_splitter.split_documents(raw_documents)

                texts = [doc.page_content for doc in documents]
                embeddings = document_embedder.embed_documents(texts)

                chunks = []
                for idx, doc in enumerate(documents):
                    chunk_id = f"{filename}#{idx}"
                    chunks.append((
                        chunk_id,
                        filename,
                        filename,
                        doc.page_content,
                        embeddings[idx],
                    ))

                _oracle_insert_chunks(chunks)
                logger.info(f"Ingested {len(chunks)} chunks from {filename}")

        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            raise ValueError(f"Failed to upload document: {e}")

    def llm_chain(
        self, query: str, chat_history: List["Message"], **kwargs
    ) -> Generator[str, None, None]:
        """Execute LLM chain without knowledge base."""

        logger.info("Using LLM without knowledge base")

        system_message = [("system", prompts.get("chat_template", "You are a helpful assistant."))]
        conversation_history = [(msg.role, msg.content) for msg in chat_history[-10:]]
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

        Retrieves relevant documents from Oracle 26ai, includes them in the prompt,
        and yields the streamed response.
        """

        logger.info("Using RAG to generate response")

        if not _vectorstore_ready:
            logger.warning("Vector store not initialized")
            return iter(["I don't have access to a knowledge base. Please upload some documents first."])

        try:
            top_k = getattr(settings.retriever, 'top_k', 4)

            if ORACLE_SEARCH_MODE == "hybrid":
                docs = _oracle_hybrid_search(query, top_k=top_k)
            else:
                query_embedding = document_embedder.embed_query(query)
                docs = _oracle_search(query_embedding, top_k=top_k)

            if not docs:
                logger.warning("No relevant documents found")
                return iter(["I couldn't find relevant information in my knowledge base for your question."])

            context_parts = []
            self._last_citations = []

            for doc in docs:
                source_label = doc["source"] or doc["filename"] or "Document"
                doc_id = doc["chunk_id"]
                content = doc["content"]
                score = doc["score"]

                context_parts.append(f"[Source: {source_label}]\n{content}")

                self._last_citations.append({
                    "document_type": "text",
                    "document_id": doc_id,
                    "document_name": source_label,
                    "content": content,
                    "metadata": str(doc),
                    "score": score,
                })

            context = "\n\n---\n\n".join(context_parts)

            rag_template = prompts.get("rag_template", """Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {input}

Answer:""")

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

            return chain.stream({"context": context, "input": query}, config={"callbacks": [self.cb_handler]})

        except Exception as e:
            logger.error(f"RAG chain error: {e}")
            return iter([f"I encountered an error while searching: {str(e)}"])

    def document_search(self, content: str, num_docs: int) -> List[Dict[str, Any]]:
        """Search for relevant documents with scores."""

        if not _vectorstore_ready:
            return []

        try:
            if ORACLE_SEARCH_MODE == "hybrid":
                docs = _oracle_hybrid_search(content, top_k=num_docs)
            else:
                query_embedding = document_embedder.embed_query(content)
                docs = _oracle_search(query_embedding, top_k=num_docs)
            return [
                {
                    "source": doc["source"] or doc["filename"] or "unknown",
                    "content": doc["content"],
                    "score": doc["score"],
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return []

    def get_documents(self) -> List[str]:
        """Get list of ingested documents."""

        if not _vectorstore_ready:
            return []

        try:
            table = ORACLE_DOCS_TABLE if ORACLE_SEARCH_MODE == "hybrid" else ORACLE_TABLE_NAME
            pool = _get_oracle_pool()
            with pool.acquire() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT DISTINCT filename FROM {table} ORDER BY filename")
                return [row[0] for row in cur]
        except Exception as e:
            logger.error(f"Get documents error: {e}")
            return []

    def delete_documents(self, filenames: List[str]) -> bool:
        """Delete documents from vector store by filename."""

        if not _vectorstore_ready:
            return False

        try:
            table = ORACLE_DOCS_TABLE if ORACLE_SEARCH_MODE == "hybrid" else ORACLE_TABLE_NAME
            pool = _get_oracle_pool()
            with pool.acquire() as conn:
                cur = conn.cursor()
                for fn in filenames:
                    cur.execute(f"DELETE FROM {table} WHERE filename = :1", [fn])
                conn.commit()
            logger.info(f"Deleted documents: {filenames}")
            return True
        except Exception as e:
            logger.error(f"Delete documents error: {e}")
            return False

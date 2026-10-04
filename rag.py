# ============================================================
# MULTIMODAL RAG - RETRIEVAL + ANSWERING
# ============================================================

import os
import io
import base64
from pathlib import Path

from PIL import Image

from groq import Groq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENV
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )


if not PINECONE_API_KEY:

    raise ValueError(
        "PINECONE_API_KEY is missing from .env"
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

TEXT_MODEL = (
    "openai/gpt-oss-20b"
)

VISION_MODEL = (
    "qwen/qwen3.8-27b"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

PINECONE_INDEX_NAME = (
    "novacore-multimodal-rag"
)

PINECONE_NAMESPACE = (
    "fy2026-demo"
)


BASE_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# 3. GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


text_llm = ChatGroq(

    model=TEXT_MODEL,

    temperature=0
)


# ============================================================
# 4. EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(

    model_name=
        EMBEDDING_MODEL,

    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# 5. CONNECT TO EXISTING PINECONE
# ============================================================

vectorstore = PineconeVectorStore(

    index_name=
        PINECONE_INDEX_NAME,

    embedding=
        embeddings,

    namespace=
        PINECONE_NAMESPACE
)


# ============================================================
# 6. RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(

    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 7. TEXT RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template(

    """
You are a helpful assistant answering questions
about the NovaCore Systems FY2026 company report.

Use ONLY the retrieved context below.

If the answer is not available in the context,
say:

"I could not find that information in the report."

Mention page numbers when possible.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""
)


text_rag_chain = (

    rag_prompt

    | text_llm

    | StrOutputParser()
)


# ============================================================
# 8. IMAGE → DATA URI
# ============================================================

def image_to_data_uri(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image.thumbnail(
        (1600, 1600)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# 9. FORMAT CONTEXT
# ============================================================

def format_context(
    retrieved_docs
):

    parts = []


    for doc in retrieved_docs:

        page = doc.metadata.get(
            "page"
        )

        modality = doc.metadata.get(
            "modality"
        )

        parts.append(

            f"[Page {page} | "
            f"{modality.upper()}]\n"
            f"{doc.page_content}\n"
        )


    return "\n\n".join(
        parts
    )


# ============================================================
# 10. VISION ANSWER
# ============================================================

def answer_with_vision(

    question,

    context,

    image_paths
):

    image_paths = image_paths[:3]


    content = [

        {

            "type": "text",

            "text": f"""

You are answering questions about the
NovaCore Systems FY2026 report.

Use ONLY the retrieved context and
the attached retrieved visuals.

RETRIEVED CONTEXT:

{context}

QUESTION:

{question}

Instructions:

- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers when possible.
- Do not invent information.
- If the information is missing, say:

"I could not find that information in the report."

"""
        }
    ]


    # ========================================================
    # ADD IMAGES
    # ========================================================

    for image_path in image_paths:

        content.append(

            {

                "type": "image_url",

                "image_url": {

                    "url":
                        image_to_data_uri(
                            image_path
                        )
                }
            }
        )


    # ========================================================
    # CALL VISION MODEL
    # ========================================================

    response = (

        groq_client

        .chat

        .completions

        .create(

            model=
                VISION_MODEL,

            messages=[

                {

                    "role":
                        "user",

                    "content":
                        content
                }
            ],

            temperature=0,

            max_completion_tokens=900
        )
    )


    return (

        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# 11. MAIN RAG FUNCTION
# ============================================================

def ask_rag(
    question
):

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    retrieved_docs = (

        retriever.invoke(
            question
        )
    )


    # --------------------------------------------------------
    # Format context
    # --------------------------------------------------------

    context = format_context(
        retrieved_docs
    )


    # --------------------------------------------------------
    # Find visual documents
    # --------------------------------------------------------

    image_paths = []


    for doc in retrieved_docs:

        if (

            doc.metadata.get(
                "modality"
            )

            == "visual"

        ):

            image_path = (

                doc.metadata.get(
                    "image_path"
                )
            )


            if (

                image_path

                and Path(
                    image_path
                ).exists()

            ):

                image_paths.append(
                    image_path
                )


    # ========================================================
    # ROUTE TO TEXT OR VISION
    # ========================================================

    if image_paths:

        answer = answer_with_vision(

            question=
                question,

            context=
                context,

            image_paths=
                image_paths
        )

    else:

        answer = text_rag_chain.invoke(

            {

                "context":
                    context,

                "question":
                    question
            }
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "answer":
            answer,

        "documents":
            retrieved_docs,

        "image_paths":
            image_paths
    }
# ============================================================
# MULTIMODAL RAG - INGESTION / INDEXING
# ============================================================
#
# Run this file when:
# 1. You are setting up the project for the first time
# 2. You change or replace the PDF
#
# Normal questions do NOT run this file.
# ============================================================

import os
import io
import time
import base64
from pathlib import Path

import fitz

from PIL import Image

from groq import Groq

from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing from .env"
    )


# ============================================================
# 2. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    BASE_DIR
    / "data"
    / "NovaCore_Multimodal_Company_Report_2026.pdf"
)

IMAGE_DIR = (
    BASE_DIR
    / "extracted_images"
)

IMAGE_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

VISION_MODEL = "qwen/qwen3.8-27b"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# PINECONE
# ============================================================

PINECONE_INDEX_NAME = (
    "novacore-multimodal-rag"
)

PINECONE_NAMESPACE = "fy2026-demo"


# ============================================================
# CHECK PDF
# ============================================================

if not PDF_PATH.exists():

    raise FileNotFoundError(
        f"""
PDF not found.

Expected:

{PDF_PATH}
"""
    )


print("=" * 70)
print("MULTIMODAL RAG - INGESTION")
print("=" * 70)

print(
    "PDF:",
    PDF_PATH
)

print()


# ============================================================
# 3. GROQ CLIENT
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# 4. IMAGE TO DATA URI
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
# 5. GENERATE VISUAL SUMMARY
# ============================================================

def summarize_visual(
    image_path,
    page_number
):

    print(
        f"  Generating visual summary "
        f"for page {page_number}..."
    )

    image_data = (
        image_to_data_uri(
            image_path
        )
    )

    prompt = f"""
This visual was extracted from page {page_number}
of the NovaCore Systems FY2026 company report.

Describe the useful business information visible
in this visual.

If it is a chart or graph:

- mention important values
- mention highest and lowest values
- mention the main trend

If it is a diagram:

- identify important components
- explain the flow
- explain important relationships

If it is a normal business image:

- describe useful factual information

Keep the description concise and factual.

Do not invent information that is not visible.
"""

    response = (
        groq_client
        .chat
        .completions
        .create(

            model=VISION_MODEL,

            messages=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",
                            "text": prompt
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],

            temperature=0,

            max_completion_tokens=500
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
# 6. EXTRACT PDF CONTENT
# ============================================================

def extract_multimodal_documents(
    pdf_path
):

    pdf = fitz.open(
        pdf_path
    )

    documents = []

    extracted_xrefs = set()


    # ========================================================
    # LOOP THROUGH PAGES
    # ========================================================

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        print()
        print(
            f"Processing page {page_number}..."
        )


        # ====================================================
        # TEXT
        # ====================================================

        text = (
            page
            .get_text("text")
            .strip()
        )

        if text:

            documents.append(

                Document(

                    page_content=text,

                    metadata={

                        "page":
                            page_number,

                        "modality":
                            "text",

                        "source":
                            pdf_path.name
                    }
                )
            )

            print(
                "  ✓ Text extracted"
            )


        # ====================================================
        # TABLES
        # ====================================================

        try:

            table_finder = (
                page.find_tables()
            )

            for table_number, table in enumerate(

                table_finder.tables,

                start=1
            ):

                df = table.to_pandas()

                table_text = (
                    df.to_markdown(
                        index=False
                    )
                )

                documents.append(

                    Document(

                        page_content=table_text,

                        metadata={

                            "page":
                                page_number,

                            "modality":
                                "table",

                            "table_number":
                                table_number,

                            "source":
                                pdf_path.name
                        }
                    )
                )

                print(
                    f"  ✓ Table {table_number} extracted"
                )


        except Exception as error:

            print(
                "  ! Table warning:",
                error
            )


        # ====================================================
        # IMAGES / CHARTS / DIAGRAMS
        # ====================================================

        for image_number, image_info in enumerate(

            page.get_images(
                full=True
            ),

            start=1
        ):

            xref = image_info[0]


            # Avoid duplicate images
            if xref in extracted_xrefs:

                continue


            extracted_xrefs.add(
                xref
            )


            try:

                image_data = (
                    pdf.extract_image(
                        xref
                    )
                )

                image_bytes = (
                    image_data["image"]
                )

                image_extension = (
                    image_data["ext"]
                )


                image_path = (

                    IMAGE_DIR
                    /
                    (
                        f"page_{page_number}"
                        f"_image_{image_number}"
                        f".{image_extension}"
                    )
                )


                image_path.write_bytes(
                    image_bytes
                )


                print(
                    f"  ✓ Image extracted: "
                    f"{image_path.name}"
                )


                # =================================================
                # VISION SUMMARY
                # =================================================

                try:

                    summary = (
                        summarize_visual(

                            image_path=
                                image_path,

                            page_number=
                                page_number
                        )
                    )


                    documents.append(

                        Document(

                            page_content=
                                summary,

                            metadata={

                                "page":
                                    page_number,

                                "modality":
                                    "visual",

                                "image_path":
                                    str(
                                        image_path
                                    ),

                                "source":
                                    pdf_path.name
                            }
                        )
                    )


                    print(
                        "  ✓ Visual summary generated"
                    )


                except Exception as error:

                    print(
                        "  ! Vision warning:",
                        error
                    )


            except Exception as error:

                print(
                    "  ! Image error:",
                    error
                )


    pdf.close()

    return documents


# ============================================================
# 7. EXTRACT EVERYTHING
# ============================================================

documents = (
    extract_multimodal_documents(
        PDF_PATH
    )
)


print()
print("=" * 70)

print(
    "EXTRACTION COMPLETE"
)

print(
    "Total documents:",
    len(documents)
)

print("=" * 70)


# ============================================================
# 8. CREATE EMBEDDINGS
# ============================================================

print()
print(
    "Loading embedding model..."
)


embeddings = HuggingFaceEmbeddings(

    model_name=
        EMBEDDING_MODEL,

    encode_kwargs={
        "normalize_embeddings": True
    }
)


embedding_dimension = len(

    embeddings.embed_query(
        "dimension check"
    )
)


print(
    "Embedding dimension:",
    embedding_dimension
)


# ============================================================
# 9. CONNECT TO PINECONE
# ============================================================

print()
print(
    "Connecting to Pinecone..."
)


pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# ============================================================
# 10. CREATE INDEX IF IT DOES NOT EXIST
# ============================================================

if not pc.has_index(
    PINECONE_INDEX_NAME
):

    print(
        "Creating Pinecone index..."
    )

    pc.create_index(

        name=
            PINECONE_INDEX_NAME,

        dimension=
            embedding_dimension,

        metric=
            "cosine",

        spec=
            ServerlessSpec(

                cloud=
                    "aws",

                region=
                    "us-east-1"
            )
    )


    while not pc.describe_index(
        PINECONE_INDEX_NAME
    ).status["ready"]:

        print(
            "Waiting for Pinecone..."
        )

        time.sleep(2)


    print(
        "Pinecone index created."
    )

else:

    print(
        "Pinecone index already exists."
    )


# ============================================================
# 11. CREATE VECTOR STORE
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
# 12. UPLOAD DOCUMENTS
# ============================================================

print()
print(
    "Uploading documents to Pinecone..."
)


vectorstore.add_documents(
    documents
)


print()
print("=" * 70)

print(
    "INGESTION COMPLETE!"
)

print("=" * 70)

print()
print(
    "Your PDF is now indexed in Pinecone."
)

print(
    "You can now run:"
)

print(
    "streamlit run app.py"
)
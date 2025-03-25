import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def extract_text_from_pdf(pdf_path: str):
    """This file will read the pdf"""
    pdf_reader = PdfReader(pdf_path)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text


def create_embeddings_from_pdf(pdf_path: str, model='text-embedding-3-large', dimensions=32):
    full_text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunk = text_splitter.split_text(full_text)

    # Create Embedding
    embeddings = OpenAIEmbeddings(model=model, dimensions=dimensions)

    vector_database = embeddings.embed_documents(text_chunk)

    return text_chunk, vector_database


def semantic_search(question, pdf_path: str):
    # Create embedding for the query - MUST match the dimensions used for documents
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
    query_embedding = embeddings.embed_query(question)

    # Get the document chunks and embeddings
    chunk, database = create_embeddings_from_pdf(pdf_path)

    # Reshape the query embedding to 2D (1, n_dimensions)
    query_embedding_2d = np.array(query_embedding).reshape(1, -1)
    database_2d = np.array(database)

    # Calculate cosine similarity
    scores = cosine_similarity(query_embedding_2d, database_2d)[0]

    # Get the top result
    index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

    # Improved output formatting
    print("\n" + "=" * 80)
    print(f"QUERY: {question}")
    print(f"SIMILARITY SCORE: {score:.3f}")
    print("-" * 80)

    # Print the relevant chunk with better formatting
    relevant_chunk = chunk[index]
    print("RELEVANT EXTRACT:")
    print("-" * 80)

    # Split into lines and print with limited width
    max_width = 80
    for paragraph in relevant_chunk.split('\n'):
        if paragraph.strip():  # Skip empty lines
            # Simple text wrapping
            words = paragraph.split()
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + len(current_line) <= max_width:
                    current_line.append(word)
                    current_length += len(word)
                else:
                    print(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                print(' '.join(current_line))
            print()  # Add empty line between paragraphs

    print("=" * 80 + "\n")


if __name__ == "__main__":
    path = '/Users/sadi_/Coding/AI Agents/AI_pdf_Reader/document.pdf'
    text_chunks, vector_embeddings = create_embeddings_from_pdf(path)

    query = 'list me all the feature name available in this document'
    semantic_search(query, path)

    # print("top matching results")
    # for result in results:
    #     print('-------')
    #     print('-------')
    #     print(result)

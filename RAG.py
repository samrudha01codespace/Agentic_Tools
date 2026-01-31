from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os
import hashlib
import pypdf
import sentence_transformers
from datetime import datetime
import re

# Configuration
DOCUMENTS_DIR = "/home/jarvis/Documents/"
VECTORSTORE_DIR = "pdf_vectorstore"
FILE_REGISTRY = "processed_files.txt"
OLLAMA_MODEL = "jarvis:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text):
    """Clean problematic text from PDFs"""
    # Remove non-UTF8 characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_file_hash(filepath):
    """Generate unique hash for file content"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_processed_files():
    """Load registry of already processed files"""
    if not os.path.exists(FILE_REGISTRY):
        return {}

    processed = {}
    with open(FILE_REGISTRY, "r", encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                filepath, filehash = line.strip().split("|")
                processed[filepath] = filehash
    return processed


def save_processed_file(filepath, filehash):
    """Update registry with new processed file"""
    with open(FILE_REGISTRY, "a", encoding='utf-8') as f:
        f.write(f"{filepath}|{filehash}\n")


def find_new_pdfs():
    """Identify new or modified PDFs that need processing"""
    processed_files = load_processed_files()
    new_files = []

    for root, _, files in os.walk(DOCUMENTS_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                current_hash = get_file_hash(full_path)

                if full_path not in processed_files or processed_files[full_path] != current_hash:
                    new_files.append((full_path, current_hash))

    return new_files


def process_new_pdfs(new_files):
    """Process newly added PDF files with text cleaning"""
    if not new_files:
        print("No new PDF files detected.")
        return []

    print(f"\nFound {len(new_files)} new/updated PDFs:")
    for file, _ in new_files:
        print(f"- {os.path.basename(file)}")

    docs = []
    for file_path, file_hash in new_files:
        try:
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()

            # Clean text in each document
            for doc in file_docs:
                doc.page_content = clean_text(doc.page_content)
                # Ensure metadata has required fields
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = 1
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path

            docs.extend(file_docs)
            save_processed_file(file_path, file_hash)
            print(f"Processed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return docs


def update_vector_store(new_chunks):
    """Update existing vector store with new documents"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        if os.path.exists(VECTORSTORE_DIR):
            vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, embeddings)

        vectorstore.save_local(VECTORSTORE_DIR)
        print(f"\nVector store updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except Exception as e:
        print(f"Vector store update failed: {str(e)}")
        return False


def initialize_qa_chain():
    """Initialize the QA chain with error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0
        )

        prompt = ChatPromptTemplate.from_template(
            """Analyze these PDF documents and provide detailed answers:
            Context: {context}
            Question: {question}
            - Reference specific page numbers when available
            - Include relevant statistics or quotes
            - If unsure, specify which aspects are unclear"""
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    except Exception as e:
        print(f"Failed to initialize QA chain: {str(e)}")
        return None


def main():
    print("=== Smart PDF RAG System ===")
    print(f"Monitoring: {DOCUMENTS_DIR} for PDF files\n")

    # 1. Process new PDFs
    new_files = find_new_pdfs()
    new_docs = process_new_pdfs(new_files)

    if new_docs:
        # 2. Split and update vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]  # Fixed regex
        )
        new_chunks = text_splitter.split_documents(new_docs)
        print(f"\nGenerated {len(new_chunks)} new text chunks")
        update_vector_store(new_chunks)

    # 3. Initialize QA system
    qa_system = initialize_qa_chain()
    if not qa_system:
        return

    print("\nSystem ready. Ask about your PDF documents (type 'quit' to exit):")
    while True:
        try:
            query = input("\nQuestion: ").strip()
            if query.lower() in ['quit', 'exit']:
                break

            result = qa_system.invoke({"query": query})

            print("\nAnswer:", result["result"])

            if result["source_documents"]:
                print("\nSource References:")
                for doc in result["source_documents"]:
                    source = os.path.basename(doc.metadata['source'])
                    page = doc.metadata.get('page', 'N/A')
                    print(f"- {source} (Page {page})")
                    print(f"  Excerpt: {doc.page_content[:120]}...")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Query Error: {str(e)}")


if __name__ == "__main__":
    main()

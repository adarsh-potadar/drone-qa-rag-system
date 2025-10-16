import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

print("\nStarting Drone RAG pipeline...\n")

# Step 1: Setup paths and folders
if not os.path.exists("data"):
    os.makedirs("data")
    print("Created 'data' folder. Add your .pdf or .txt drone documents there.")

# Step 2: Load documents (PDF + TXT)
print("Loading and splitting documents...")
docs = []

# Load PDF files
pdf_files = glob.glob("data/*drone_manual.pdf")
for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())

# Load Text files
txt_files = glob.glob("data/*.txt")
for file_path in txt_files:
    loader = TextLoader(file_path)
    docs.extend(loader.load())

if not docs:
    print("No documents found in 'data' folder. Please add PDF or TXT files and rerun.")
    exit()
else:
    print(f"Loaded {len(docs)} document chunks from {len(pdf_files)} PDF(s) and {len(txt_files)} TXT file(s).")

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} text chunks.")

# Step 4: Create embeddings and FAISS vectorstore
print("Creating FAISS vectorstore using HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local("vectorstore")
print("Vectorstore created and saved successfully.\n")

# Step 5: Setup Ollama model (Phi)
print("Setting up QA chain using Ollama (phi model)...")

llm = Ollama(model="phi")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
print("QA Chain is ready with Ollama + Phi model!\n")

# Step 6: Ask Questions
print("RAG Pipeline built successfully!\n")

while True:
    question = input("Ask a question about your drone (or type 'exit' to quit): ")
    if question.lower() in ["exit", "quit"]:
        print("Exiting Drone RAG Assistant.")
        break

    result = qa_chain.invoke({"query": question})
    print("\nAnswer:\n", result["result"])
    print("-" * 80)

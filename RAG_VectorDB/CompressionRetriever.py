from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from LLMProvider import llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv() 

embeddings = NVIDIAEmbeddings(
  model="nvidia/nemotron-3-embed-1b",
  api_key=os.getenv("NVIDIA_CHAT_API_KEY")
)

pdf_Loader = PyPDFLoader("Advanced_Python_FastAPI_Guide.pdf")
documents = pdf_Loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

#it takes list of documents and creates a vectorstore from them. It also persists the vectorstore to disk for later use.
vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="./chroma_db")

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

results = compression_retriever.invoke("what is comprehension?")

for doc in results:
    print(f" - {doc.page_content[:60]}")
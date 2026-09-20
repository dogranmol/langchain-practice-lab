from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
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


index_name = "first-index"  # Replace with your Pinecone index name, in server

vectorstore = PineconeVectorStore.from_documents(split_docs, embedding=embeddings, index_name=index_name)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})




prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:

    Context:
    {context}

    Question:
    {question}
""")

# Helper function
def format_docs(docs):
    print("preparing context for the question...")
    return "\n".join([doc.page_content for doc in docs])

# in chain every step get .invoked() one after another. The output of one step is passed as input to the next step.
# so rag_chain invoke goes to retriever.invoke() first, then the output of retriever.invoke() goes to RunnableLambda(format_docs).invoke(), 
#then the output of that goes to prompt_template.invoke(), then the output of that goes to llm.invoke(), and finally the output of that goes to StrOutputParser().invoke() 
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    result = rag_chain.invoke(question)
    print("Answer:", result)


# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index("my_index")
# # Query directly to pinecone
# query_vector = embeddings.embed_query("Explain FastAPI routing")
# results = index.query(vector=query_vector, top_k=3)
# print(results)

# reload index and use in langchain
# vectorstore = PineconeVectorStore(
#     index_name="my_index",
#     embedding=embeddings
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


    
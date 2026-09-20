
class HybridRetriever:

    def __init__(self, vectorstore, bm25_retriever, k:int =3):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.k = k
    
    def invoke(self, query:str) -> List[Document]:
        # Retrieve documents from vectorstore
        vector_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Retrieve documents from BM25 retriever
        bm25_docs = self.bm25_retriever.invoke(query)

        seen = set()
        combined=[]

        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                combined.append(doc)
                seen.add(doc.page_content)
        return combined
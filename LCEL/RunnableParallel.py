
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLMProvider import llm
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

prompt = ChatPromptTemplate.from_template("Summarize in 2 sentences: {question}")

chain1 = prompt | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_template("Extract three keywords from: {question}")

chain2 = prompt | llm | StrOutputParser()

#parallel execution: both chains run concurently
parallel_chain = RunnableParallel({
    "summary":chain1,
    "keywords":chain2
})

#Pass-through original text for reference i.e. text passes as input
full_chain = RunnableParallel({
    "original":RunnablePassthrough(),
    "summary":chain1,
    "keywords":chain2
})

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    result = full_chain.invoke({"question": question})
    print("Original:", result["original"])
    print("Summary:", result["summary"])
    print("Keywords:", result["keywords"])




from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLMProvider import llm

prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer: {question}")

chain = prompt | llm | StrOutputParser()


while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    result = chain.invoke({"question": question})
    print(result)


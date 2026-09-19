
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLMProvider import llm
from langchain_core.runnables import RunnableBranch

prompt = ChatPromptTemplate.from_template("You are Math expert. Solve: {question}")

math_chain = prompt | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_template("You are python expert. Write code for: {question}")

code_chain = prompt | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_template("Answer: {question}")

default_chain = prompt | llm | StrOutputParser()

#route based on presence of keywords
def route_to_chain(input_dict):
    question = input_dict.get("question", "").lower()
    if "math" in question or "calculate" in question:
        return "math"
    elif "code" in question or "python" in question:
        return "code"
    else:
        return "default"

#create routing chain
router = RunnableBranch(
    (lambda x: "math" == route_to_chain(x), math_chain),
    (lambda x: "code" == route_to_chain(x), code_chain),
    default_chain
)





while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    result = router.invoke({"question": question})
    print("Result:", result)



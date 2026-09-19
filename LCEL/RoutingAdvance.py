
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLMProvider import llm
from langchain_core.runnables import RunnableLambda

prompt = ChatPromptTemplate.from_template("You are Math expert. Solve: {question}")

math_chain = prompt | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_template("You are python expert. Write code for: {question}")

code_chain = prompt | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_template("Answer: {question}")

default_chain = prompt | llm | StrOutputParser()
from langchain_core.runnables import RunnableLambda 
 
# Use LLM to classify input 
classifier_prompt = ChatPromptTemplate.from_template( 
    "Classify as MATH, CODE, or GENERAL: {question}\n\n" 
    "Respond with only the category." 
) 
classifier = classifier_prompt | llm | StrOutputParser() 
 
def classify_and_route(input_dict): 
    """Classify question and route accordingly.""" 
    category = classifier.invoke(input_dict).strip().upper() 
     
    if category == "MATH": 
        return math_chain.invoke(input_dict) 
    elif category == "CODE": 
        return code_chain.invoke(input_dict) 
    else: 
        return default_chain.invoke(input_dict) 
 
router_with_classification = RunnableLambda(classify_and_route) 
 
result = router_with_classification.invoke({"question": "Calculate 15 * 7"}) 
print(result) 
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
import json
from langchain_core.prompts import ChatPromptTemplate
from LLMProvider import llm

strict_json_parser = JsonOutputParser()

fallback_chain = ChatPromptTemplate.from_template("You are JSON repair expert. Fix this invalid JSON: {error_text}. Return a valid json") | llm | StrOutputParser()

def safe_json_parser(input_text):
    try:
        return json.load(input_text)
    except Exception as e:
        print(f"Error parsing JSON: {e}. Attempting to fix the input using fallback chain.")
        error_text = str(e)
        fixed_json = fallback_chain.invoke({"error_text": input_text})
        return json.loads(fixed_json)
    
parser_with_fallback = RunnableLambda(safe_json_parser)

malformed_input = '{"name": "John", "age": 30, "city": "New York"'
result = parser_with_fallback.invoke(malformed_input)
print(result)
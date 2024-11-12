import math
from typing import Annotated, Sequence
import os

from IPython.display import Image, display
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
import numexpr
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from flask import Flask, request, jsonify
from utils import doc_reader
app = Flask(__name__)

# BASIC SCORER
###########################################################################################

@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )

llm = AzureChatOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0.1,
    max_tokens=None,
    timeout=None,
)
tools = [calculator]
# Remove the tool_choice parameter for older API versions
llm_with_tools = llm.bind_tools(tools)

class ChainState(TypedDict):
    """LangGraph state."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ground_truth: str
    score: bool
    response: str

def call_chain(state: ChainState, config: RunnableConfig):
    response = llm_with_tools.invoke(state["messages"], config)
    ground_truth = state["ground_truth"]
    return {"messages": [response],"ground_truth":ground_truth,}

def call_model(state: ChainState, config: RunnableConfig):
    response = llm.invoke(state["messages"], config)
    ground_truth = state["ground_truth"]
    return {"messages": [response],"ground_truth":ground_truth,}

def clean_response(state: ChainState, config: RunnableConfig) -> str:
    question = state['messages'][0].content
    last_message = state["messages"][-1]
    ground_truth = state["ground_truth"]
    prompt_str = """
    Give only the final answer without re-stating the question.
    \nResponse:\n{messages}\n
    Question:\n{question}\n
    """

    prompt_template = PromptTemplate(
        template=prompt_str,
        input_variables=["messages"],
    )

    chain = prompt_template | llm | StrOutputParser()

    response = chain.invoke({'question':question,'messages':last_message})

    return {"response":state['messages'][-1].content, "messages": [response],"ground_truth":ground_truth}

def llm_score(state: ChainState, config: RunnableConfig) -> str:
    """Use the LLM to score the text and question.
    QA:\n---{query}\n---\n
    """
    prompt_str = """
    Read the following question and answer. Return True if the ground truth is mentioned in the model answer or False if it is not. 
    Decimal places or significant figures can be different.
    Ground truth: {ground_truth}\n
    Model answer: {answer}\n
    """

    prompt_template = PromptTemplate(
        template=prompt_str,
        input_variables=["answer", "ground_truth"],
    )

    ground_truth = state["ground_truth"]
    # print(state['messages'][1].content)
    query = state['messages'][1].content
    answer = state['response']

    chain = prompt_template | llm | StrOutputParser()

    score = chain.invoke({
        # 'query':query,
        'answer':answer,
        'ground_truth':ground_truth
        })

    return {
        "response":state['response'],
        "messages": state['messages'],
        "ground_truth":ground_truth, 
        'score':score
        }

graph_builder = StateGraph(ChainState)
graph_builder.add_node("call_tool", call_chain)
graph_builder.add_node("execute_tool", ToolNode(tools))
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("clean_response", clean_response)
graph_builder.add_node("llm_score", llm_score)
graph_builder.set_entry_point("call_tool")
graph_builder.add_edge("call_tool", "execute_tool")
graph_builder.add_edge("execute_tool", "call_model")
graph_builder.add_edge("call_model", "clean_response")
graph_builder.add_edge("clean_response", 'llm_score')
graph_builder.add_edge('llm_score',END)
chain = graph_builder.compile()

def qa_maths_reasoning_langgraph(text: str, question: str, ground_truth):
    query = f"""read the following text:\n---{text}\n---\nQuestion: {question}"""
    result = chain.invoke({'messages': ['user', query], 'ground_truth': ground_truth})
    return result

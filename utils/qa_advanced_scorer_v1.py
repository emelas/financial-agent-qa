import math
import numexpr
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from IPython.display import Image, display
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
import os

@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.
    If proportion or portion is mentioned, give the answer as a percentage.

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
    temperature=0,
    max_tokens=None,
    timeout=None,
)
tools = [calculator]
# Remove the tool_choice parameter for older API versions
llm_with_tools = llm.bind_tools(tools,)

class ChainState(TypedDict):
    """LangGraph state with enhanced scoring."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ground_truth: str
    score_reasoning: str
    detailed_score: dict
    response: str

def call_chain(state: ChainState, config: RunnableConfig):
    # print(f'call_chain state: {state}')
    response = llm_with_tools.invoke(state["messages"], config)
    # print(f'call_chain response: {response} ({type(response)})')
    ground_truth = state["ground_truth"]
    return {"messages": [response],"ground_truth":ground_truth,}

def call_model(state: ChainState, config: RunnableConfig):
    # prompt = hub.pull("hwchase17/structured-chat-agent")
    # agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent,
    #     tools=tools,
    #     verbose=False,
    #     handle_parsing_errors=True,  # Handle any parsing errors gracefully
    # )
    # response = agent_executor.invoke({'input':state},)['input']['messages'][0]
    response = llm.invoke(state["messages"], config)
    print(f'call_model response: {response}')
    ground_truth = state["ground_truth"]
    return {"messages": [response],"ground_truth":ground_truth,}

def clean_response(state: ChainState, config: RunnableConfig) -> str:
    question = state['messages'][0].content
    last_message = state["messages"][-1]
    ground_truth = state["ground_truth"]
    prompt_str = """
    Give only the final answer with its unit where relevant without re-stating the question.
    \nResponse:\n{messages}\n
    Question:\n{question}\n
    """

    prompt_template = PromptTemplate(
        template=prompt_str,
        input_variables=["messages"],
    )

    chain = prompt_template | llm | StrOutputParser()

    response = chain.invoke({'question':question,'messages':last_message})

    return {"response":response, "messages": [response],"ground_truth":ground_truth}

def llm_score(state: ChainState, config: RunnableConfig) -> str:
    """Use the LLM to score the text and question.
    """

    prompt_str = """
    Compare the numerical values in the ground truth and model answer to decide if they are the same answer.
    Follow these steps:
    1. Extract the numerical value from both answers. Ignore the units.
    2. Return True if the absolute values are equivalent within a reasonable margin, False otherwise
    
    Ground truth: {ground_truth}
    Model answer: {answer}
    
    Let's solve this step by step:
    1. Ground truth number: [extract number]
    2. Compare with tolerance: [comparison result]
    
    Final answer (True/False): """

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
        'score_reasoning':score
        }

def advanced_scorer(state: ChainState, config: RunnableConfig) -> str:
    """Advanced scoring agent that considers multiple aspects of the answer."""
    
    prompt_str = """
    Perform a detailed analysis of the model's answer compared to the ground truth.
    Consider multiple aspects in your evaluation:
    
    1. Numerical Accuracy:
       - Extract and compare numerical values, ignoring percentage signs for example
       - Consider acceptable margin of error (±1% or correct to the whole number)
       - Check for unit consistency
       - If the absolute values of the ground truth and answer match closely (within ±1%) but have opposite signs due to a decline or increase context, consider them equivalent.    
    
    2. Conceptual Correctness:
       - Verify if the approach/methodology is correct
       - Check if all required components are present
    
    3. Context Relevance:
       - Ensure the answer addresses the specific question
       - Verify if any contextual requirements are met
    
    Question: {query}
    Ground truth: {ground_truth}
    Model answer: {answer}
    
    Analyze the response and return a JSON object with this exact structure:
    {{
        "numerical_accuracy": {{
            "score": <float between 0 and 1>,
            "reasoning": "<explanation>"
        }},
        "conceptual_correctness": {{
            "score": <float between 0 and 1>,
            "reasoning": "<explanation>"
        }},
        "context_relevance": {{
            "score": <float between 0 and 1>,
            "reasoning": "<explanation>"
        }},
        "overall_score": <float between 0 and 1>,
        "is_correct": <boolean>
    }}
    
    Ensure your response is a valid JSON object matching this structure exactly.
    """

    prompt_template = PromptTemplate(
        template=prompt_str,
        input_variables=["query", "answer", "ground_truth"],
    )

    chain = prompt_template | llm | JsonOutputParser()

    detailed_score = chain.invoke({
        'query': state['messages'][0].content,
        'answer': state['response'],
        'ground_truth': state['ground_truth']
    })

    return {
        "response": state['response'],
        "messages": state['messages'],
        "ground_truth": state['ground_truth'],
        "detailed_score": detailed_score
    }

# Update the graph with new scoring nodes
graph_builder = StateGraph(ChainState)
graph_builder.add_node("call_tool", call_chain)
graph_builder.add_node("execute_tool", ToolNode(tools))
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("clean_response", clean_response)
graph_builder.add_node("llm_score", llm_score)
graph_builder.add_node("advanced_scorer", advanced_scorer)
# graph_builder.add_node("consensus_scorer", consensus_scorer)
graph_builder.set_entry_point("call_tool")

# Define the enhanced flow
graph_builder.add_edge("call_tool", "execute_tool")
graph_builder.add_edge("execute_tool", "call_model")
graph_builder.add_edge("call_model", "clean_response")
graph_builder.add_edge("clean_response", "llm_score")
graph_builder.add_edge("llm_score", "advanced_scorer")
graph_builder.add_edge("advanced_scorer", END)
# graph_builder.add_edge("advanced_scorer", "consensus_scorer")
# graph_builder.add_edge("consensus_scorer", END)

chain = graph_builder.compile()

def qa_maths_reasoning_langgraph_advanced_scorer(text: str, question: str, ground_truth):
    query = f"""read the following text:\n---{text}\n---\nQuestion: {question}"""
    result = chain.invoke({'messages': ['user', query], 'ground_truth': ground_truth})
    result['question'] = question
    result['overall_score'] = result['detailed_score']['overall_score']
    result
    return result

# display(Image(chain.get_graph().draw_mermaid_png()))
# result = qa_maths_reasoning_langgraph(text, question, answer)
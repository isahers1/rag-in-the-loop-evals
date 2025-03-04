from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import StateGraph

from agent.configuration import Configuration
from agent.state import State
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import uuid

mini = ChatOpenAI(model="gpt-4o-mini")
full = ChatOpenAI(model="gpt-4o")

models = {
    "mini": mini,
    "4o": full,
}

RETRIEVER_EVAL_PROMPT = """
You are an expert data labeler evaluating outputs for relevance to the input. Your task is to assign a score based on the following rubric:

<Rubric>
Retrieved outputs that are relevant:
- Contain information that could help answer the input
- Can also contain superfluous information, but it should still be somewhat related
</Rubric>

<Instruction>
- Read and understand the full meaning of the input (including edge cases)
- Formulate a list of facts that would be needed to respond to the input
- Identify all the information contained in the outputs
- Match each fact needed to answer the input with the information in the outputs
- Note any facts that are not found
- Note any outputs that are completely irrelevant
</Instruction>

<Reminder>
Focus solely on whether the information in the outputs can answer the input. Explain thoroughly why this is or isn't the case. Make sure to use partial credit.
</Reminder>

<input>
{input}
</input>

<outputs>
{outputs}
</outputs>
"""

retriever_evaluator = create_llm_as_judge(
    prompt=RETRIEVER_EVAL_PROMPT,
    continuous=True
)

GENERATOR_EVAL_PROMPT = """
You are an expert data labeler evaluating model outputs for relevance to some context. Your task is to assign a score based on the following rubric:

<Rubric>
Relevant outputs:
- Only contains basic facts (i.e. the sky is blue) and information from the context
- Do not stray too far from the information contained in the context
- Do not hallucinate information
</Rubric>

<Instruction>
- Read the context
- Construct a list of all the facts/opinions in the context
- Extract all the facts/opinions from the model output
- Cross reference both these lists
- Penalize any facts/opinions in the model output that are not basic facts or mentioned in the context
</Instruction>

<outputs>
{outputs}
</outputs>

<context>
{context}
</context>
"""

generator_evaluator = create_llm_as_judge(
    prompt=GENERATOR_EVAL_PROMPT,
    continuous=True
)


class SearchQueries(BaseModel):
    """Queries to search for based on user question."""
    queries: list[str] = Field(...,description="A list of 3-5 queries that cover the users question. The queries should be diverse.")

def query_generator(state: State, config: RunnableConfig):
    """Generate queries"""
    model_name = config["configurable"].get("model", "mini")
    model = models[model_name]
    query_llm = model.with_structured_output(SearchQueries, include_raw=True)
    if len(state['messages']) > 0 and state['messages'][-1].role == "tool":
        query_messages = [HumanMessage(content=f"Please generate search queries for {state['question']}. Remember last time you generated the following queries: {state['queries']}, which resulted in the following feedback: {state['messages'][-1].content}")]
    else:
        query_messages = [HumanMessage(content=f"Please generate search queries for {state['question']}")]
    response = query_llm.invoke(query_messages)
    return {"queries": response['parsed'].queries, "messages": query_messages + [response['raw']]}

def retrieval_node(state: State):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("./src/agent/local_faiss_index", embeddings, allow_dangerous_deserialization=True)
    context = ""
    for query in state['queries']:
        retrieved_docs = vector_store.similarity_search(query, k=2)
        context += '\n'.join([doc.page_content for doc in retrieved_docs])
    return {"context": context}

def eval_retrieval_node(state: State, config: RunnableConfig):
    do_eval = config["configurable"].get("do_eval", False)
    if not do_eval:
        return Command(goto="generate_answer")
    else:
        eval_res = retriever_evaluator(inputs=state['question'], outputs=state['context'])
        if eval_res['score'] >= 0.8:
            return Command(goto="__end__")
        else:
            tool_call_id = uuid.uuid4()
            return Command(
                update = {"messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "retriever_evaluator_tool",
                                    "args": {},
                                    "id": tool_call_id, 
                                    "type": "tool_call"
                                }
                            ]
                        ),
                        ToolMessage(
                            content=eval_res['comment'],
                            tool_call_id=tool_call_id
                        )
                    ]
                },
                goto="query_generator"
            )

def generate_answer(state: State, config: RunnableConfig):
    model_name = config["configurable"].get("model", "mini")
    model = models[model_name]
    if state['messages'][-1].type == "tool":
        generate_messages = [HumanMessage(content=f"Please use the retrieved context to answer the question. <context>{state['context']}</context>\n\n<question>{state['question']}</question>. Here is some feedback on a previous generation: {state['messages'][-1].content}")]
    else:
        generate_messages = [HumanMessage(content=f"Please use the retrieved context to answer the question. <context>{state['context']}</context>\n\n<question>{state['question']}</question>")]
    
    response = model.invoke(generate_messages)
    return {"messages": generate_messages + [response]}

def eval_generation_node(state: State, config: RunnableConfig):
    do_eval = config["configurable"].get("do_eval", False)
    if not do_eval:
        return Command(goto="__end__")
    else:
        eval_res = generator_evaluator(outputs=state['messages'][-1].content,context=state['context'])
        if eval_res['score'] >= 0.8:
            return Command(goto="__end__")
        else:
            tool_call_id = uuid.uuid4()
            return Command(
                update = {"messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "generator_evaluator_tool",
                                    "args": {},
                                    "id": tool_call_id, 
                                    "type": "tool_call"
                                }
                            ]
                        ),
                        ToolMessage(
                            content=eval_res['comment'],
                            tool_call_id=tool_call_id
                        )
                    ]
                },
                goto="generate_answer"
            )


# Define a new graph
workflow = StateGraph(State, config_schema=Configuration)

# Add the node to the graph
workflow.add_node(query_generator)
workflow.add_node(retrieval_node)
workflow.add_node(eval_retrieval_node)
workflow.add_node(generate_answer)
workflow.add_node(eval_generation_node)


workflow.add_edge("__start__", "query_generator")
workflow.add_edge("query_generator", "retrieval_node")
workflow.add_edge("retrieval_node", "eval_retrieval_node")
workflow.add_edge("generate_answer", "eval_generation_node")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "RAG with in-the-loop evals"  # This defines the custom name in LangSmith

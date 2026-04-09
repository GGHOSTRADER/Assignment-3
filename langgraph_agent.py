# agent
import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES
from llm_utils import safe_invoke


# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)

# 🔴 ADDED
SOURCE_LABELS = {
    "apple": "Apple 10-K",
    "tesla": "Tesla 10-K",
    "FY24_Q4_Consolidated_Financial_Statements": "Apple 10-K",
    "tsla-20241231-gen": "Tesla 10-K",
}


# 🔴 ADDED
def get_friendly_source_name(doc, fallback_key: str) -> str:
    raw_source = doc.metadata.get("source", fallback_key)
    return SOURCE_LABELS.get(raw_source, SOURCE_LABELS.get(fallback_key, raw_source))


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}

    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(
                persist_directory=persist_dir, embedding_function=embeddings
            )
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
            continue

    return retrievers


RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    needs_rewrite: str


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- 🔍 RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()

    # --- [START] Improved Routing Logic ---
    options = ["apple", "tesla", "both", "none"]
    router_prompt = f"""
Route the user question to the correct datasource.

Valid options:
{', '.join(options)}

Return ONLY valid JSON in exactly this format:
{{"datasource":"apple"}}

Rules:
- Choose apple if the question is only mentions apple.
- Choose tesla if the question is only mentions tesla.
- Choose "both" if the question compares or involves apple and tesla.
- Choose "none" if neither entity is present in options.
- Do not include markdown.
- Do not include explanation.
- Do not include extra keys.

User Question: {question}
"""
    try:
        response = safe_invoke(llm, router_prompt)
        content = response.content.strip()
        # Handle cases where LLM might wrap JSON in backticks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        res_json = json.loads(content)
        target = res_json.get("datasource", "both")
    except Exception as e:
        print(
            colored(
                f"⚠️ Error parsing router output: {e}. Defaulting to 'both'.", "yellow"
            )
        )
        target = "both"

    print(colored(f"🎯 Routing to: {target}", "cyan"))
    # --- [END] ---

    docs_content = ""
    targets_to_search = []
    if target == "both":
        targets_to_search = list(FILES.keys())
    elif target in FILES:
        targets_to_search = [target]

    for t in targets_to_search:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)
            for d in docs:
                # 🟢 CHANGED
                source_name = get_friendly_source_name(d, t)
                docs_content += f"\n\n[Source: {source_name}]\n{d.page_content}"

    return {"documents": docs_content, "search_count": state["search_count"] + 1}


@retry_logic
def grade_documents_node(state: AgentState):
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    system_prompt = """You are a grader assessing relevance. 
    
    - Your Goal: Idenfity if the retrieved document contain information related to the user question?
    
    ***CRITICAL:*** You must answer with ONLY one word: 'yes' or 'no'. Do not add any explanation."""

    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Retrieved document context: \n\n {documents} \n\n User question: {question}"
        ),
    ]

    response = safe_invoke(llm, msg)
    content = response.content.strip().lower()

    grade = "yes" if "yes" in content else "no"
    print(f"   Relevance Grade: {grade}")
    return {"needs_rewrite": grade}


@retry_logic
def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial analyst.\n"
                "Answer ONLY using the provided context.\n"
                "If the exact answer is not in the context, say 'I don't know'.\n"
                "ALWAYS cite the source in brackets, e.g. [Source: Apple 10-k].\n\n"
                "IMPORTANT RULES:\n"
                "- Distinguish carefully between 3-month and 12-month values.\n"
                "- If the question asks about 2024 annual results, use the 12-month / full-year 2024 figure, not the quarterly figure.\n"
                "- Do not guess or infer from percentages if the exact figure is present.\n"
                "- If multiple values appear, explain which one matches the question.\n"
                "- If the value appears clearly in narrative text (not a table), you MUST extract it.\n"
                "- Do NOT restrict yourself to tables only.\n"
                "- Narrative sentences containing explicit financial figures are valid sources.\n"
                "Examples of valid extraction:\n"
                "Table:\n"
                "Row: banana sales | 2022 | 495\n"
                "Question: What is banana co banana sales revenue in 2022?\n"
                "Answer: In 2022, banana co's banana sales revenue was $495 million [Source: banana co 10-K].\n\n"
                "Narrative in text:\n"
                "Costs amounted to $1.54 billion in 2023\n"
                "Question: What were ford costs in 2023?\n"
                "Answer: In 2024, Ford's cost were $1.54 billion [Source: ford 10-K].\n"
                "Both are valid.\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm
    response = safe_invoke(chain, {"context": documents, "question": question})
    return {"generation": response.content}


@retry_logic
def rewrite_node(state: AgentState):
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()

    msg = [
        HumanMessage(
            content=f"The previous search for this question did not retrieve useful results:\n"
            f"{question}\n\n"
            f"Rewrite the question ONLY to improve document retrieval.\n"
            f"DO NOT change the meaning.\n"
            f"DO NOT add new years, forecasts, projections, guidance, or extra topics.\n"
            f"DO NOT broaden the scope.\n"
            f"Keep the company name, metric, and year if present.\n"
            f"Output ONLY the rewritten query text.\n\n"
            f"Example:\n"
            f"Question: Tesla 2024 年的研發費用是多少？\n"
            f"Output: Tesla 2024 research and development expenses"
        )
    ]
    response = safe_invoke(llm, msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        if state["needs_rewrite"] == "yes":
            return "generate"
        else:
            if state["search_count"] > 2:
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_graph_agent(question: str):
    app = build_graph()
    inputs = {
        "question": question,
        "search_count": 0,
        "needs_rewrite": "no",
        "documents": "",
        "generation": "",
    }
    # Using stream to see progress if needed, but invoke is fine for simple return
    result = app.invoke(inputs)
    return result["generation"]


# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(
            create_retriever_tool(
                retriever,
                f"search_{key}_financials",
                f"Searches {key.capitalize()}'s financial data.",
            )
        )

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: think step-by-step about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

IMPORTANT:
- When you have found the exact answer, you MUST stop calling tools.
- Immediately produce:
  Thought: I now know the final answer
  Final Answer: ...

- DO NOT repeat the same search if the answer is already visible.
- DO NOT loop. Maximum 1–2 searches unless absolutely necessary.
- NEVER output "Action: None". If no action is needed, go directly to Final Answer.

Begin!

Question: {input}
Thought: {agent_scratchpad}

***CONSTRAINTS***
- English Only: Final Answer must be in English.
- Year Precision: Carefully select the 2024 column (not 2023/2022).
- Honesty: If the exact 2024 figure is not found, say "I don't know".
- Prefer tables over narrative text when extracting numbers.
"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    try:
        result = safe_invoke(agent_executor, {"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"

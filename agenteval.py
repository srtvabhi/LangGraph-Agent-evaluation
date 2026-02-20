"""
Agent evaluation with LangGraph + LangSmith.

Covers three evaluation styles:
1) Final response quality.
2) Full trajectory (all tool calls).
3) Single-step tool selection quality.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from langsmith import Client
from langsmith.evaluation import evaluate
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Environment requirements
# -----------------------------------------------------------------------------
# Required env vars:
# - OPENAI_API_KEY
# - LANGSMITH_API_KEY
# - LANGSMITH_TRACING=true
# - LANGSMITH_PROJECT=<project-name>
load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(2 + 3) * 4'."""
    # Very restricted characters for safety.
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expression):
        return "Invalid expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"


@tool
def weather(city: str) -> str:
    """Return a mocked weather report for a city."""
    weather_map = {
        "new york": "Cloudy, 12C",
        "san francisco": "Sunny, 18C",
        "london": "Rainy, 9C",
    }
    return weather_map.get(city.lower().strip(), "Unknown, 20C")


TOOLS = [calculator, weather]


class AgentState(MessagesState):
    pass


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def call_model(state: AgentState) -> AgentState:
    system = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use tools when needed and keep answers concise."
        )
    )
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
agent_app = graph.compile()


@dataclass
class AgentRunOutput:
    final_response: str
    tool_trajectory: List[Dict[str, Any]]
    first_tool_call: Optional[Dict[str, Any]]


def run_agent(prompt: str) -> AgentRunOutput:
    result = agent_app.invoke({"messages": [HumanMessage(content=prompt)]})
    msgs = result["messages"]

    final_response = ""
    trajectory: List[Dict[str, Any]] = []
    first_tool_call: Optional[Dict[str, Any]] = None

    for msg in msgs:
        if isinstance(msg, AIMessage):
            if msg.content:
                final_response = str(msg.content)
            for tc in msg.tool_calls or []:
                call = {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id"),
                }
                trajectory.append(call)
                if first_tool_call is None:
                    first_tool_call = call

    return AgentRunOutput(
        final_response=final_response,
        tool_trajectory=trajectory,
        first_tool_call=first_tool_call,
    )


def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = inputs["prompt"]
    out = run_agent(prompt)
    return {
        "final_response": out.final_response,
        "tool_trajectory": out.tool_trajectory,
        "first_tool_call": out.first_tool_call,
    }


def final_response_evaluator(
    outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    pred = outputs.get("final_response", "").strip().lower()
    ref = reference_outputs.get("expected_final_contains", "").strip().lower()
    score = 1.0 if ref and ref in pred else 0.0
    return {
        "key": "final_response_correct",
        "score": score,
        "comment": f"Expected final response to contain: '{ref}'",
    }


def trajectory_evaluator(
    outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    pred_names = [x.get("name") for x in outputs.get("tool_trajectory", [])]
    exp_names = reference_outputs.get("expected_tool_sequence", [])
    score = 1.0 if pred_names == exp_names else 0.0
    return {
        "key": "trajectory_exact_match",
        "score": score,
        "comment": f"Predicted: {pred_names}, expected: {exp_names}",
    }


def single_step_evaluator(
    outputs: Dict[str, Any], reference_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    pred = (outputs.get("first_tool_call") or {}).get("name")
    exp = reference_outputs.get("expected_first_tool")
    score = 1.0 if pred == exp else 0.0
    return {
        "key": "single_step_first_tool",
        "score": score,
        "comment": f"Predicted first tool: {pred}, expected: {exp}",
    }


def ensure_dataset(client: Client, dataset_name: str) -> str:
    existing = next((d for d in client.list_datasets(dataset_name=dataset_name)), None)
    if existing:
        return existing.id

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Agent evaluation dataset for final response, trajectory, and single-step checks.",
    )

    examples = [
        {
            "inputs": {"prompt": "What is (8 + 4) * 2?"},
            "outputs": {
                "expected_final_contains": "24",
                "expected_tool_sequence": ["calculator"],
                "expected_first_tool": "calculator",
            },
        },
        {
            "inputs": {"prompt": "What's the weather in London?"},
            "outputs": {
                "expected_final_contains": "rainy",
                "expected_tool_sequence": ["weather"],
                "expected_first_tool": "weather",
            },
        },
        {
            "inputs": {"prompt": "Say hello to me without using any tools."},
            "outputs": {
                "expected_final_contains": "hello",
                "expected_tool_sequence": [],
                "expected_first_tool": None,
            },
        },
    ]

    for ex in examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs=ex["inputs"],
            outputs=ex["outputs"],
        )

    return dataset.id


def main() -> None:
    dataset_name = os.getenv("LANGSMITH_DATASET", "agent-eval-demo-dataset")
    experiment_prefix = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", "agent-eval-demo")

    client = Client()
    dataset_id = ensure_dataset(client, dataset_name=dataset_name)

    results = evaluate(
        target,
        data=dataset_id,
        evaluators=[
            final_response_evaluator,
            trajectory_evaluator,
            single_step_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        max_concurrency=4,
    )

    print("Evaluation started. View run summary in LangSmith.")
    print(f"Dataset: {dataset_name}")
    print(f"Experiment prefix: {experiment_prefix}")
    print(f"Results object: {results}")


if __name__ == "__main__":
    main()

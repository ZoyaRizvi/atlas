from __future__ import annotations
import operator
from typing import Annotated, TypedDict, Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from atlas.phase4.agent_tools import ATLAS_TOOLS


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def make_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    return agent_node


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_agent_graph(
    model: str = "claude-sonnet-4-6",
    system_prompt: str = "",
    on_step: Callable[[str, str], None] | None = None,
):
    # Build the LLM with tools
    llm = ChatAnthropic(model=model, max_tokens=2048)
    llm_with_tools = llm.bind_tools(ATLAS_TOOLS)

    # Wrap the agent node to emit step callbacks
    raw_agent_node = make_agent_node(llm_with_tools)

    def instrumented_agent_node(state: AgentState) -> dict:
        result = raw_agent_node(state)
        if on_step:
            msg = result["messages"][-1]
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # Agent decided to call a tool — report what it chose
                    for tc in msg.tool_calls:
                        args_str = str(tc.get("args", {}))[:120]
                        on_step("tool_call", "%s(%s)" % (tc["name"], args_str))
                else:
                    # Agent produced a final answer
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    on_step("final_answer", content[:200])
        return result

    # Wrap ToolNode to emit step callbacks for tool results
    raw_tool_node = ToolNode(ATLAS_TOOLS)

    def instrumented_tool_node(state: AgentState) -> dict:
        result = raw_tool_node(state)
        if on_step:
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage):
                    snippet = str(msg.content)[:150].replace("\n", " ")
                    on_step("tool_result", "[%s] %s" % (msg.name, snippet))
        return result

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", instrumented_agent_node)
    graph.add_node("tools", instrumented_tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()

def run_agent(
    question: str,
    model: str = "claude-sonnet-4-6",
    system_prompt: str = "You are Atlas, a helpful research assistant. "
                         "Use tools when they help answer the question accurately. "
                         "Always show your reasoning before calling a tool.",
    on_step: Callable[[str, str], None] | None = None,
) -> tuple[str, list[dict]]:
    
    trace: list[dict] = []

    def collect_step(step_type: str, content: str):
        trace.append({"type": step_type, "content": content})
        if on_step:
            on_step(step_type, content)

    messages = []
    if system_prompt:
        from langchain_core.messages import SystemMessage
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=question))

    compiled = build_agent_graph(
        model=model,
        system_prompt=system_prompt,
        on_step=collect_step,
    )

    result = compiled.invoke({"messages": messages})

    # Extract the final text answer from the last AIMessage
    final_answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return final_answer, trace
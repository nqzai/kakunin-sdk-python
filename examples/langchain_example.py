"""
LangChain integration example for Kakunin SDK.

This example shows how to use KakuninToolGuard to enforce
agent scope limits on a LangChain tool before execution.
"""

import asyncio
import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from kakunin import Kakunin
from kakunin.integrations.langchain import KakuninToolGuard

# Load API keys from environment
KAKUNIN_API_KEY = os.environ.get("KAKUNIN_API_KEY", "kak_live_...")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


async def main() -> None:
    # Step 1 — Connect to Kakunin
    async with Kakunin(api_key=KAKUNIN_API_KEY) as client:

        # Step 2 — Register an agent
        agent = await client.agents.create(
            name="ResearchBot-1",
            model="llama-3.1-8b-instant",
            version="2025-01",
        )
        print(f"Agent registered: {agent.id}")

        # Step 3 — Define a LangChain tool
        @tool
        def search_web(query: str) -> str:
            """Search the web for information about a topic."""
            # Replace with real search API in production
            return f"Search results for: {query}"

        @tool
        def summarize_text(text: str) -> str:
            """Summarize a given piece of text."""
            # Replace with real summarization in production
            return f"Summary: {text[:100]}..."

        # Step 4 — Wrap tools with KakuninToolGuard
        # Guard enforces scope before every tool call
        guarded_search = KakuninToolGuard(
            kakunin=client,
            agent_id=agent.id,
            tool=search_web,
            required_scopes=["web.search"],
        )

        guarded_summarize = KakuninToolGuard(
            kakunin=client,
            agent_id=agent.id,
            tool=summarize_text,
            required_scopes=["text.summarize"],
        )

        # Step 5 — Build LangChain agent with guarded tools
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful research assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        tools = [guarded_search, guarded_summarize]
        langchain_agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=langchain_agent, tools=tools)

        # Step 6 — Run the agent
        result = executor.invoke({
            "input": "Search for LangChain RAG and summarize the results."
        })
        print(f"\nAgent result: {result['output']}")

        # Step 7 — Record a behavioral event
        event = await client.events.create(
            agent_id=agent.id,
            action_type="research_completed",
            details={"query": "LangChain RAG"},
        )
        print(f"Risk band: {event.risk_band}")


if __name__ == "__main__":
    asyncio.run(main())
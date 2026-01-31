import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

# LangChain Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# --- Configuration ---
CONFIG = {
    "API_KEY": "AIzaSyBrlD4z5wBEcY_W2-CmfwFGY_HNKF252Xo",  # Replace with your real key
    "CSE_ID": "9204a16ddb91246f5"  # Replace with your real CSE ID
}


def fetch_page_text(url: str) -> str:
    """Fetches main textual content from a webpage, with error handling."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/1.0; +http://yourdomain.com/bot.html)"
        }
        res = requests.get(url, headers=headers, timeout=7)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # Try to find main content blocks with common tags/classes/ids
        main_content_elements = soup.find_all(
            ['article', 'main', 'div'],
            class_=['main-content', 'article-content', 'post-content', 'entry-content', 'body-content'],
            id=['main', 'content', 'article', 'body']
        )
        content_parts = []
        if main_content_elements:
            for element in main_content_elements:
                paragraphs = element.find_all("p")
                list_items = element.find_all('li')
                text_paragraphs = "\n".join(
                    p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20
                )
                text_list = "\n".join(
                    li.get_text(strip=True) for li in list_items if len(li.get_text(strip=True)) > 10
                )
                if text_paragraphs:
                    content_parts.append(text_paragraphs)
                if text_list and text_list not in text_paragraphs:
                    content_parts.append(text_list)
        else:
            # Fallback: take all paragraphs and list items
            paragraphs = soup.find_all("p")
            list_items = soup.find_all('li')
            text_paragraphs = "\n".join(
                p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20
            )
            text_list = "\n".join(
                li.get_text(strip=True) for li in list_items if len(li.get_text(strip=True)) > 10
            )
            if text_paragraphs:
                content_parts.append(text_paragraphs)
            if text_list and text_list not in text_paragraphs:
                content_parts.append(text_list)

        combined_text = "\n\n".join(content_parts).strip()
        if combined_text:
            return combined_text[:6000]  # Limit length for LLM input
        else:
            return "[No substantial text content found on page]"

    except requests.exceptions.RequestException as req_err:
        return f"[Failed to fetch content from {url}: Network error - {req_err}]"
    except Exception as e:
        return f"[Failed to fetch content from {url}: Parsing error - {e}]"


def run_deep_research_agent(query: str) -> str:
    llm_model = ChatOllama(model="jarvis:latest")
    print(f"\n🚀 Deep Research Agent initiated for: '{query}'")
    try:
        search_service = build("customsearch", "v1", developerKey=CONFIG["API_KEY"])

        # Fetch up to 50 results in batches of 10
        all_items = []
        for start_index in range(1, 50, 10):  # Google allows up to 10 per call
            search_res = search_service.cse().list(
                q=query,
                cx=CONFIG["CSE_ID"],
                num=10,
                start=start_index
            ).execute()

            items = search_res.get("items", [])
            if not items:
                break  # No more results
            all_items.extend(items)

        if not all_items:
            return "🚫 No relevant search results found."

        collected_contents = []
        source_links = []

        for item in all_items:
            link = item.get("link")
            if link and link not in source_links:
                print(f"  Fetching content from: {link}")
                content = fetch_page_text(link)
                if content and not content.startswith("[Failed to fetch content"):
                    collected_contents.append(f"--- Document from {link} ---\n{content}")
                    source_links.append(link)
                else:
                    print(f"Skipped {link} due to fetch issues.")

        if not collected_contents:
            return "🚫 No useful content could be retrieved from search results."

        combined_text = "\n\n".join(collected_contents)

        messages = [
            SystemMessage(content="You are a helpful assistant synthesizing research documents."),
            HumanMessage(content=f"""
You are an expert researcher. Using the following documents, answer the query: "{query}".

--- Documents ---
{combined_text}
--- End of Documents ---

Please synthesize a concise, accurate, well-structured report with sources.
""")
        ]

        synthesized_report = llm_model.invoke(messages).content
        final_report = f"{synthesized_report}\n\nSources:\n" + "\n".join(source_links)

        return final_report

    except Exception as e:
        return f"❌ Deep Research Agent failed unexpectedly: {e}"


class DeepResearchInput(BaseModel):
    query: str = Field(description="The detailed research query to perform.")


# Inner Deep Research Agent setup

deep_research_llm = ChatOllama(model="jarvis:latest", temperature=0.0)

deep_research_tools = [
    Tool(
        name="WebFetcher",
        func=fetch_page_text,
        description="Fetches full textual content from a given webpage URL."
    )
]

deep_research_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a 'Deep Research' AI. Use the 'WebFetcher' tool to gather webpage content for a query.
Follow this format STRICTLY:

Thought: ...
Action: [one of {tool_names}]
Action Input: ...
Observation: ...
... repeat Thought/Action/Action Input/Observation as needed ...
Final Answer: [Your final synthesized research report, no tool calls or ReAct formatting]

Available tools:
{tools}
"""),
    ("user", "{input}"),
    ("ai", "{agent_scratchpad}"),
])

deep_research_agent_executor = AgentExecutor(
    agent=create_react_agent(deep_research_llm, deep_research_tools, deep_research_prompt),
    tools=deep_research_tools,
    verbose=True,
    handle_parsing_errors=True
)

# Master Agent Setup

master_llm = ChatOllama(model="jarvis:latest")

master_tools = [
    Tool(
        name="DeepResearchAgent",
        func=run_deep_research_agent,
        description="Perform comprehensive web research and synthesize detailed reports.",
        args_schema=DeepResearchInput
    )
]

master_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a general-purpose AI assistant with access to the 'DeepResearchAgent' tool.

Follow this format STRICTLY:

Thought: ...
Action: [one of {tool_names}]
Action Input: ...
Observation: ...
... repeat as needed ...
Final Answer: [Your final synthesized research report, no tool calls or ReAct formatting]


Available tools:
{tools}
"""),
    ("human", "{input}"),
    ("user", "{agent_scratchpad}"),
])

master_agent_executor = AgentExecutor(
    agent=create_react_agent(master_llm, master_tools, master_prompt),
    tools=master_tools,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    query = "Jarvis, please give me a list of extra-curricular classes specifically for women available in Gandhinagar on Saturdays and Sundays. Include the class name, type (e.g., dance, yoga, coding), timing, location, and contact details if possible."

    print(f"\n--- Running Master Agent with Query: '{query}' ---")

    try:
        response = master_agent_executor.invoke({"input": query, "agent_scratchpad": []})
        print("\n--- Master Agent Final Response ---\n")
        print(response['output'])
    except Exception as e:
        print(f"\nError during master agent execution: {e}")
        print("Check Ollama server, model names, API keys, and ReAct formatting compatibility.")

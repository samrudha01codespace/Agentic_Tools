import atexit
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Union

from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import Tool

import alarm_agent
import calendar_agent
from Tools import *

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Configuration
CONFIG = {
    "API_KEY": "AIzaSyDNiyz26F8067FFaJBBGsZLtAClw9g73gc",
    "CSE_ID": "9204a16ddb91246f5",
    "HISTORY_FILE": "chat_history.json",
    "AUTO_SAVE": True,
    "WEATHER_API_KEY": "569f582ffbd28dc7fc880d27a6b3493b"
}
ACCUWEATHER_API_KEY = "QL2Gi0peE3zn8UHrjwsbWPFDGJ0ctF35"

BASE_URL = "http://api.openweathermap.org/data/2.5"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/direct"

SYSTEM_PROMPT = """

You are **Jarvis**, a highly intelligent AI assistant. You serve the user with precision, addressing them as *"Sir"* in all interactions. Your responses are **concise, direct, and human-like**, avoiding unnecessary elaboration. Prioritize brevity while maintaining accuracy and context-awareness.  

### **Response Format Rules:**  
1. **Always greet the user as *"Sir"*** (e.g., *"Sir, the weather is clear today."*).  
2. **If no tool is required**, respond with JSON: {"final_answer": "your response"}  
3. **If a tool is needed**, respond with JSON: {"thought": "reasoning", "action": "tool name", "action_input": "input"}  

### **Strict Output Format:**  

Always respond with a valid JSON object as described above. Nothing else.

### **Key Guidelines:**  
- **Be brief but insightful.** No rambling.  
- **Tools are optional.** Only use if critical.  
- **Maintain a professional, respectful tone.**  
- **Never omit "Sir" in replies.**  

### **Available Tools:**  
{tools}  

### **Example Interaction:**  
For "What’s the time in London?"  
{"thought": "Sir needs the current time in London.", "action": "WorldClock", "action_input": "London"}  

For "hi"  
{"final_answer": "Sir, hello! How can I assist you today?"}


<important notes>
1. Follow the instructions carefully and do not deviate from them. Do not ask for confirmations for any additional information
2. Always output valid JSON.
3. DO NOT REPEAT YOUR STEPS. STOP PROCESSING ONCE THE DRAFT IS CREATED.
</important notes>  

  """


class RobustOutputParser(JsonOutputParser):
    """Enhanced output parser with better error handling."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            text = text.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)

            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if isinstance(data, dict):
                        if "final_answer" in data:
                            return AgentFinish({"output": data["final_answer"]}, text)
                        elif "action" in data and "action_input" in data:
                            return AgentAction(data["action"], data["action_input"], text)
                except json.JSONDecodeError:
                    pass

            # Fallback for non-JSON responses
            if "Final Answer:" in text:
                return AgentFinish({"output": text.split("Final Answer:")[-1].strip()}, text)

            return AgentFinish({"output": text}, text)
        except Exception as e:
            return AgentFinish({"output": f"Error parsing response: {str(e)}"}, text)


def clean_content(content: Any) -> str:
    """Sanitize content for storage and display."""
    if isinstance(content, str):
        content = re.sub(r'Could not parse LLM output.*?Observation:\s*', '', content)
        content = re.sub(r'For troubleshooting.*?OUTPUT_PARSING_FAILURE', '', content)
        return content.strip()
    return str(content)


def animate_text(text: str, prefix: str = "") -> None:
    """Display text with typewriter effect."""
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.02)
    print()


def realtime() -> str:
    return current_time


def show_thinking_animation() -> None:
    """Visual indicator for processing."""
    for _ in range(3):
        for frame in ["   ", ".  ", ".. ", "..."]:
            sys.stdout.write(f"\rThinking{frame}")
            sys.stdout.flush()
            time.sleep(0.2)
    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()




def save_history(memory: ConversationBufferMemory) -> None:
    """Persist conversation history to disk."""
    try:
        history = {
            "messages": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": clean_content(msg.content),
                    "timestamp": datetime.now().isoformat()
                }
                for msg in memory.chat_memory.messages
            ]
        }
        with open(CONFIG["HISTORY_FILE"], 'w') as f:
            json.dump(history, f, indent=2)
        print("\n[System] History saved successfully")
    except Exception as e:
        print(f"\n[System] History save failed: {str(e)}")


def load_history(memory: ConversationBufferMemory) -> bool:
    """Load conversation history from disk."""
    if not os.path.exists(CONFIG["HISTORY_FILE"]):
        return False

    try:
        with open(CONFIG["HISTORY_FILE"], 'r') as f:
            history = json.load(f)

        memory.clear()
        for msg in history.get("messages", []):
            if msg["type"] == "human":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])

        print("\n[System] History loaded successfully")
        return True
    except Exception as e:
        print(f"\n[System] History load failed: {str(e)}")
        return False


def initialize_agent_system() -> AgentExecutor:
    """Configure and initialize the AI agent system."""
    tools = [
        Tool(
            name="WebSearch",
            func=web_search,
            description="Search the web for current or trending information. Input: your query."
        ),
        Tool(
            name="CurrentWeather",
            func=get_current_weather,
            description="Get current weather info for any city. Input: city name."
        ),
        Tool(
            name="Forecast5Day3Hour",
            func=get_forecast_5day,
            description="Get 5-day weather forecast in 3-hour intervals. Input: city name."
        ),
        Tool(
            name="AirPollution",
            func=get_air_pollution,
            description="Get current air pollution data for any city. Input: city name."
        ),
        Tool(
            name="OneCallSummary",
            func=get_onecall_summary,
            description="Get summary from One Call API (current, hourly, and daily forecast). Input: city name."
        ),
        Tool(
            name="SystemStatus",
            func=get_system_status,
            description="Check system performance (CPU, RAM, disk). No input needed."
        ),
        Tool(
            name="PDFDocuments",
            func=document_loaded,
            description="Ask specific questions about uploaded PDF documents."
        ),
        Tool(
            name="Alarm",
            func=alarm_agent.set_alarm,
            description="Set an alarm. Format: 'Set alarm for YYYY-MM-DD HH:MM:SS with message YourMessage'."
        ),
        Tool(
            name="Calendar Events",
            func=calendar_agent.calendar_tool,
            description="Manage events. Input: 'Create event' or 'List events'."
        ),
        Tool(
            name="Terminal",
            func=execute_terminal_command,
            description="Run Linux terminal commands. Input: a valid shell command. This function is to control the PC."
        ),
        Tool(
            name="Coder",
            func=coder,
            description="Generate code. Input format: <|Language|> followed by your coding request."
        ),
        Tool(
            name="ImageAnalyzer",
            func=analyze_with_ollama,
            description="Analyze the content of an image using a natural language query. Input format: your question about the image (e.g., 'What is in this image? <Image Path>')."
        ),
        Tool(
            name="AccuCurrentWeather",
            func=accuweather_current_weather,
            description="Get current weather conditions using AccuWeather. Input: city name."
        ),
        Tool(
            name="AccuForecast5Day",
            func=accuweather_forecast_5day,
            description="Get a 5-day weather forecast from AccuWeather. Input: city name."
        ),
        Tool(
            name="AccuHourlyForecast12Hr",
            func=accuweather_hourly_12hr,
            description="Get 12-hour hourly forecast using AccuWeather. Input: city name."
        ),
        Tool(
            name="AccuAirQuality",
            func=accuweather_air_quality,
            description="Get air quality data from AccuWeather. Input: city name."
        ),
        Tool(
            name="AccuGeoPositionSearch",
            func=accuweather_geoposition_search,
            description="Convert coordinates into location details using AccuWeather Geoposition Search. Input: 'lat,lon'."
        ),
        Tool(
            name="AccuLocationSearch",
            func=accuweather_location_search,
            description="Search for a location key using city name with AccuWeather. Input: city name."
        ),
        Tool(
            name="TerminalCommands",
            func=terminal_codes,
            description="Get terminal commands using a natural language query. For Example: If Tony Starks tells you to perform some tasks in terminal and you don't know the command then ask here."
        ),
        Tool(
            name="SSHCommandExecutor",
            func=ssh_command_with_password,
            description="Execute a remote command over SSH using username and password. This SSH function will connect the Tony Stark's Mobile to control "
                        "Input: remote_command."
        ),
        Tool(
            name="ExtractPDFTextTool",
            func=extract_text_from_pdfs,
            description=(
                "Extracts and returns text from a list of PDF file paths. "
                "Automatically uses OCR for scanned or image-based pages. "
                "Input: List of PDF paths (e.g., ['/path/to/file1.pdf', '/path/to/file2.pdf'])."
            )
        ),
        Tool(
            name="JarvisDeepThink",
            func=think_and_answer,
            description="Thoughtfully answers complex questions using the DeepSeek LLM, mimicking Jarvis-style deep reasoning and logical deduction."
        ),
        Tool(
            name="DeepSearch",
            func=deepsearch,
            description="Performs an in-depth, multi-step reasoning and search process. Ideal for answering complex queries that require logical deduction, cross-domain analysis, and contextual understanding."
        )

    ]

    llm_instance = LlamaCpp(
        model_path="/Users/nic/Desktop/llama-3.2-1b-instruct-q8_0.gguf",
        n_ctx=4096,  # Increased from 512 to 4096
        n_batch=512,  # Standard batching
        f16_kv=True,  # High-precision Key-Value cache
        n_gpu_layers=-1,  # Maximum Metal acceleration
        verbose=False  # Reduces console noise during operation
    )

    memory = ConversationSummaryBufferMemory(
        llm=ChatOllama(model="gemma2:2b"),
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=8000
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    return initialize_agent(
        tools=tools,
        llm=ChatOllama(model="gemma2:2b"),
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        agent_kwargs={
            "prompt": prompt,
            "memory_key": "chat_history"
        },
        output_parser=RobustOutputParser(),
        handle_parsing_errors=True
    )



def main() -> None:


    """Main execution loop for the assistant system."""
    agent = initialize_agent_system()
    load_history(agent.memory)

    if CONFIG["AUTO_SAVE"]:
        atexit.register(save_history, agent.memory)

    print("\n" + "═" * 60)
    print("JARVIS Assistant Initialized".center(60))
    print("Type '/exit' to quit or '/help' for commands".center(60))
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("\n You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit"):
                break
            elif user_input.lower() == "/save":
                save_history(agent.memory)
                continue
            elif user_input.lower() == "/load":
                load_history(agent.memory)
                continue
            elif user_input.lower() == "/help":
                print("\nAvailable commands:\n"
                      "/exit - Quit the program\n"
                      "/save - Manually save conversation history\n"
                      "/load - Reload previous conversation\n"
                      "/help - Show this help message")
                continue
            elif user_input.lower() == "jarvis":
                user_input = input("You: ").strip()
                if not user_input:
                    continue

            start_response_time = time.time()
            agent.memory.chat_memory.add_user_message(user_input)
            show_thinking_animation()

            response = agent.invoke({"input": user_input})
            end_response_time = time.time()
            agent.memory.chat_memory.add_ai_message(response.get("output", response))
            response_time = end_response_time - start_response_time

            print(response.get("output", response))
            print(f"[Response Time] {response_time:.2f} seconds")

            if CONFIG["AUTO_SAVE"] and len(agent.memory.chat_memory.messages) % 3 == 0:
                save_history(agent.memory)

        except KeyboardInterrupt:
            print("\n[System] Session interrupted")
            break
        except Exception as e:
            print(f"\n[System] Error: {str(e)}")
            continue

    if CONFIG["AUTO_SAVE"]:
        save_history(agent.memory)
    print("\n[System] Session ended successfully")


if __name__ == "__main__":
    main()

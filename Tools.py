import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Dict, Any

import ollama
import pdfplumber
import psutil
import pygame
import pytesseract
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from gtts import gTTS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import RAG
from DeepSearch import master_agent_executor
from main import show_thinking_animation, SYSTEM_PROMPT

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


"""AccuWeather API"""

ACCUWEATHER_BASE_URL = "http://dataservice.accuweather.com"


def accuweather_location_search(city_name: str) -> str:
    url = f"{ACCUWEATHER_BASE_URL}/locations/v1/cities/search"
    params = {"apikey": ACCUWEATHER_API_KEY, "q": city_name}
    response = requests.get(url, params=params)
    data = response.json()
    if data:
        return data[0]["Key"]
    else:
        raise Exception("Location key not found.")


def accuweather_geoposition_search(coords: str) -> str:
    lat, lon = coords.split(',')
    url = f"{ACCUWEATHER_BASE_URL}/locations/v1/cities/geoposition/search"
    params = {"apikey": ACCUWEATHER_API_KEY, "q": f"{lat},{lon}"}
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("LocalizedName", "Location not found")


def accuweather_current_weather(location: str) -> str:
    try:
        location_key = accuweather_location_search(location)
        url = f"{ACCUWEATHER_BASE_URL}/currentconditions/v1/{location_key}"
        params = {"apikey": ACCUWEATHER_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()[0]
        return (f"📍 {location} Weather:"

                f"🌡️ Temp: {data['Temperature']['Metric']['Value']}°C"

                f"☁️ Condition: {data['WeatherText']}")
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def accuweather_forecast_5day(location: str) -> str:
    try:
        location_key = accuweather_location_search(location)
        url = f"{ACCUWEATHER_BASE_URL}/forecasts/v1/daily/5day/{location_key}"
        params = {"apikey": ACCUWEATHER_API_KEY, "metric": "true"}
        response = requests.get(url, params=params)
        forecast_data = response.json()["DailyForecasts"]
        result = f"📅 5-Day Forecast for {location}:"

        for day in forecast_data:
            date = day['Date'].split('T')[0]
            min_temp = day['Temperature']['Minimum']['Value']
            max_temp = day['Temperature']['Maximum']['Value']
            desc = day['Day']['IconPhrase']
            result += f"{date} | {min_temp}-{max_temp}°C | {desc}"

        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def accuweather_hourly_12hr(location: str) -> str:
    try:
        location_key = accuweather_location_search(location)
        url = f"{ACCUWEATHER_BASE_URL}/forecasts/v1/hourly/12hour/{location_key}"
        params = {"apikey": ACCUWEATHER_API_KEY, "metric": "true"}
        response = requests.get(url, params=params)
        hourly_data = response.json()
        result = f"⏰ 12-Hour Forecast for {location}:"

        for hour in hourly_data:
            time = hour['DateTime'].split('T')[1][:5]
            temp = hour['Temperature']['Value']
            condition = hour['IconPhrase']
            result += f"{time} | {temp}°C | {condition}"

        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def accuweather_air_quality(location: str) -> str:
    try:
        location_key = accuweather_location_search(location)
        url = f"{ACCUWEATHER_BASE_URL}/currentconditions/v1/{location_key}/airquality"
        params = {"apikey": ACCUWEATHER_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()[0]['AirAndPollen']
        result = f"🌫️ Air Quality in {location}:"
        for item in data:
            if item['Category']:
                result += f"{item['Name']}: {item['Category']}"
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def web_search(query: str) -> str:
    """Perform a web search using Google Custom Search API."""
    print(f"\n🔍 Searching: '{query}'...")
    show_thinking_animation()

    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": query,
                "key": CONFIG["API_KEY"],
                "cx": CONFIG["CSE_ID"]
            },
            timeout=5
        )
        results = response.json().get("items", [])

        if not results:
            return "No relevant results found."

        best = results[0]
        meta = best.get("pagemap", {}).get("metatags", [{}])[0]
        description = meta.get("og:description", meta.get("description", "No description available"))

        return (f"**{best.get('title', 'Unknown')}**\n"
                f"🔗 {best.get('link', 'No link')}\n"
                f"📄 {description}")
    except Exception as e:
        return f"Search failed: {str(e)}"


def fetch_page_text(url: str) -> str:
    """
    Fetches a webpage and extracts its main textual content.
    Tries to focus on common article/main content areas to get cleaner text.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; JarvisBot/1.0; +http://www.example.com/bot.html)"}
        res = requests.get(url, headers=headers, timeout=7)
        res.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        soup = BeautifulSoup(res.text, "html.parser")

        # Prioritize common article/main content tags
        # This is a heuristic and might need adjustment for specific websites.
        main_content_elements = soup.find_all(['article', 'main', 'div'],
                                              class_=['main-content', 'article-content', 'post-content',
                                                      'entry-content'],
                                              id=['main', 'content', 'article'])

        # If specific main content elements are found, try to extract from them
        content_parts = []
        if main_content_elements:
            for element in main_content_elements:
                paragraphs = element.find_all("p")
                list_items = element.find_all(['li'])

                # Filter out very short or empty paragraphs/list items
                text_from_paragraphs = "\n".join(
                    p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
                text_from_list_items = "\n".join(
                    li.get_text(strip=True) for li in list_items if len(li.get_text(strip=True)) > 10)

                if text_from_paragraphs:
                    content_parts.append(text_from_paragraphs)
                if text_from_list_items and text_from_list_items not in text_from_paragraphs:  # Avoid duplicates
                    content_parts.append(text_from_list_items)

        # Fallback: if no specific main content elements found, extract all paragraphs and lists from the body
        if not content_parts:
            paragraphs = soup.find_all("p")
            list_items = soup.find_all(['li'])

            text_from_paragraphs = "\n".join(
                p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            text_from_list_items = "\n".join(
                li.get_text(strip=True) for li in list_items if len(li.get_text(strip=True)) > 10)

            if text_from_paragraphs:
                content_parts.append(text_from_paragraphs)
            if text_from_list_items and text_from_list_items not in text_from_paragraphs:
                content_parts.append(text_from_list_items)

        combined_text = "\n\n".join(content_parts).strip()

        # Return cleaned text, limited to avoid exceeding LLM context windows
        return combined_text[:6000] if combined_text else "[No substantial text content found on page]"

    except requests.exceptions.RequestException as req_err:
        return f"[Failed to fetch content from {url}: Network error - {req_err}]"
    except Exception as e:
        return f"[Failed to fetch content from {url}: Parsing error - {e}]"


def deep_web_research(query: str, num_results=5) -> str:
    """
    Performs deep web research by first searching, then fetching full content
    from relevant URLs, and finally compiling a report of raw content.
    The final synthesis is left to the calling LLM agent.
    """
    print(f"\n🔍 Deep research initiated for: '{query}'")
    # show_thinking_animation() # Assuming this is a utility function you have

    try:
        # Step 1: Use Google Custom Search API to get initial results (URLs and snippets)
        search_service = build("customsearch", "v1", developerKey=CONFIG["API_KEY"])
        search_res = search_service.cse().list(q=query, cx=CONFIG["CSE_ID"], num=num_results).execute()

        items = search_res.get("items", [])
        if not items:
            return "🚫 No relevant search results found by Google Custom Search."

        # Initialize the report with a clear header
        report_parts = [f"--- Deep Research Report for '{query}' ---"]
        source_links = []  # To keep track of sources for citation

        # Step 2: Iterate through search results, fetch content, and add to report
        for idx, item in enumerate(items):
            title = item.get("title", "Untitled")
            link = item.get("link")
            snippet = item.get("snippet", "No snippet available.")

            # Skip duplicate links or non-HTTP/HTTPS links
            if not link or not (link.startswith('http://') or link.startswith('https://')) or link in source_links:
                continue

            report_parts.append(f"\n## {idx + 1}. {title}\n🔗 {link}\n")
            report_parts.append(f"**Snippet:** {snippet}\n")

            # Fetch full page text
            page_text = fetch_page_text(link)
            report_parts.append(f"**Full Content Excerpt:**\n```text\n{page_text}\n```\n")
            source_links.append(link)  # Add link to sources list after attempting fetch

        if not source_links:  # Check if any content was successfully fetched
            return "🚫 No useful content could be retrieved after attempting to fetch from search results."

        # Add a section for all collected source URLs at the end
        report_parts.append("\n--- All Source URLs ---")
        report_parts.append("\n".join(source_links))

        # Join all parts of the report for the LLM
        return "\n".join(report_parts)

    except requests.exceptions.RequestException as req_err:
        return f"❌ Deep research failed due to a network error: {req_err}"
    except Exception as e:
        return f"❌ Deep research failed unexpectedly: {e}"


def get_coordinates(location: str) -> tuple[Any, Any]:
    response = requests.get(GEOCODING_URL, params={
        "q": location,
        "limit": 1,
        "appid": CONFIG["WEATHER_API_KEY"]
    })
    data = response.json()
    if data:
        return data[0]["lat"], data[0]["lon"]
    else:
        raise Exception("Location not found.")


def get_current_weather(location: str) -> str:
    try:
        response = requests.get(f"{BASE_URL}/weather", params={
            "q": location,
            "appid": CONFIG["WEATHER_API_KEY"],
            "units": "metric"
        })
        if response.status_code == 200:
            data = response.json()
            main = data['main']
            weather = data['weather'][0]
            return (f"📍 Weather in {location}:\n"
                    f"🌡️ Temperature: {main['temp']}°C\n"
                    f"☁️ Condition: {weather['description'].capitalize()}\n"
                    f"💧 Humidity: {main['humidity']}%\n"
                    f"🌬️ Wind: {data['wind']['speed']} m/s")
        return f"❌ Weather info not found for {location}"
    except Exception as e:
        return f"⚠️ Weather API error: {str(e)}"


def get_forecast_5day(location: str) -> str:
    try:
        response = requests.get(f"{BASE_URL}/forecast", params={
            "q": location,
            "appid": CONFIG["WEATHER_API_KEY"],
            "units": "metric"
        })
        if response.status_code == 200:
            data = response.json()['list'][:20]  # Show just first 5 entries (3-hour intervals)
            forecast_str = f"📅 5-Day/3-Hour Forecast for {location}:\n"
            for entry in data:
                dt = datetime.fromtimestamp(entry['dt']).strftime("%Y-%m-%d %H:%M")
                temp = entry['main']['temp']
                desc = entry['weather'][0]['description'].capitalize()
                forecast_str += f"{dt} | {temp}°C | {desc}\n"
            return forecast_str
        return f"❌ Forecast not found for {location}"
    except Exception as e:
        return f"⚠️ Forecast error: {str(e)}"


def get_air_pollution(location: str) -> str:
    try:
        lat, lon = get_coordinates(location)
        response = requests.get(AIR_POLLUTION_URL, params={
            "lat": lat,
            "lon": lon,
            "appid": CONFIG["WEATHER_API_KEY"]
        })
        if response.status_code == 200:
            data = response.json()['list'][0]['components']
            return (f"🌫️ Air Pollution in {location}:\n"
                    f"CO: {data['co']} µg/m3\n"
                    f"NO: {data['no']} µg/m3\n"
                    f"NO₂: {data['no2']} µg/m3\n"
                    f"O₃: {data['o3']} µg/m3\n"
                    f"SO₂: {data['so2']} µg/m3\n"
                    f"PM2.5: {data['pm2_5']} µg/m3\n"
                    f"PM10: {data['pm10']} µg/m3")
        return f"❌ Air pollution data not found for {location}"
    except Exception as e:
        return f"⚠️ Air Pollution API error: {str(e)}"


def get_onecall_summary(location: str) -> str:
    try:
        lat, lon = get_coordinates(location)
        response = requests.get(ONECALL_URL, params={
            "lat": lat,
            "lon": lon,
            "appid": CONFIG["WEATHER_API_KEY"],
            "units": "metric",
            "exclude": "minutely"
        })
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            hourly = data['hourly'][:3]
            daily = data['daily'][:2]

            summary = (f"🧭 OneCall Summary for {location}:\n"
                       f"🌡️ Now: {current['temp']}°C, {current['weather'][0]['description'].capitalize()}\n"
                       f"🔮 Next Hours:\n")
            for h in hourly:
                time = datetime.fromtimestamp(h['dt']).strftime("%H:%M")
                summary += f"{time} - {h['temp']}°C, {h['weather'][0]['main']}\n"
            summary += f"📆 Next Days:\n"
            for d in daily:
                date = datetime.fromtimestamp(d['dt']).strftime("%Y-%m-%d")
                summary += f"{date} - {d['temp']['min']}°C to {d['temp']['max']}°C, {d['weather'][0]['main']}\n"
            return summary
        return f"❌ OneCall data not found"
    except Exception as e:
        return f"⚠️ OneCall API error: {str(e)}"


# def email_loaded(to, subject, body) -> str:
#     try:
#         agent = AgentEmail
#         service = agent.authenticate_gmail_api()
#         result = agent.send_message(service, to, subject, body)
#
#         # Format response nicely
#         response = {
#             "answer": result.get("result", "No result"),
#             "emails": [
#                 {
#                     "subject": email.get("subject", "No Subject"),
#                     "from": email.get("from", "Unknown"),
#                     "snippet": email.get("snippet", "")[:100] + "...",
#                     "date": email.get("date", "Unknown")
#                 }
#                 for email in result.get("emails", [])
#             ] if result.get("emails") else []
#         }
#         return json.dumps(response)
#     except Exception as e:
#         return json.dumps({"error": f"Error querying emails: {str(e)}"})


def get_system_status(query: str) -> Dict[str, Any]:
    """Collect and analyze system performance metrics."""
    status = {
        "OS": platform.system(),
        "Version": platform.version(),
        "CPU": psutil.cpu_percent(interval=1),
        "Memory": psutil.virtual_memory().percent,
        "Disk": psutil.disk_usage('/').percent,
        "Uptime": time.time() - psutil.boot_time(),
        "Battery": psutil.sensors_battery().percent if hasattr(psutil, "sensors_battery") else "N/A",
        "Recommendations": []
    }

    if status["CPU"] > 80:
        status["Recommendations"].append("High CPU usage - close unnecessary applications")
    if status["Memory"] > 85:
        status["Recommendations"].append("High memory usage - clear caches")
    if status["Disk"] > 90:
        status["Recommendations"].append("Low disk space - clean temporary files")
    if status["Battery"] != "N/A" and status["Battery"] < 20:
        status["Recommendations"].append("Low battery - connect to power")

    return status


APP_LOG_FILE = "used_apps_log.json"

def get_recent_app_usage(limit: int = 5) -> Dict[str, Any]:
    """
    Fetch the most recently used GUI applications with metadata.
    Designed to be used by the LLM.
    """
    if not os.path.exists(APP_LOG_FILE):
        return {
            "error": f"Log file '{APP_LOG_FILE}' not found.",
            "apps": []
        }

    try:
        with open(APP_LOG_FILE, "r") as f:
            history = json.load(f)

        # Sort by end_time, descending
        recent_apps = sorted(history, key=lambda x: x.get("end_time", ""), reverse=True)
        limited_apps = recent_apps[:limit]

        return {
            "apps": limited_apps,
            "count": len(limited_apps),
            "source": APP_LOG_FILE
        }

    except Exception as e:
        return {
            "error": str(e),
            "apps": []
        }


def execute_terminal_command(command: str) -> Dict[str, Any]:
    """Execute a shell command and return output and errors."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10  # prevent infinite loops
        )
        return {
            "command": command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "stdout": "", "stderr": "Command timed out", "returncode": -1}
    except Exception as e:
        return {"command": command, "stdout": "", "stderr": str(e), "returncode": -1}


def play_audio(text: str) -> None:
    # Convert text to speech
    tts = gTTS(text=text, lang='en')

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)

    # Optionally delete the file after playing
    os.remove(temp_path)


def general(query: str) -> str:
    response = ollama.generate(model="jarvis:latest", query=query)
    return response['response']['text']


def document_loaded(query: str) -> str:
    try:
        qa_system = RAG.initialize_qa_chain()
        result = qa_system({"query": query})

        # Format the response in a way that's easier to parse
        response = {
            "answer": result["result"],
            "sources": [
                {
                    "source": os.path.basename(doc.metadata['source']),
                    "page": doc.metadata.get('page', 'N/A'),
                    "excerpt": doc.page_content[:120] + "..."
                }
                for doc in result["source_documents"]
            ] if result.get("source_documents") else []
        }
        return json.dumps(response)  # Return as JSON string for better parsing
    except Exception as e:
        return json.dumps({"error": f"Error querying documents: {str(e)}"})


def text_to_speech(text: str, lang: str = "en") -> None:
    """Convert text to speech and play it."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("output.mp3")
        if platform.system() == "Windows":
            os.system("start output.mp3")
        else:
            os.system("mpg321 output.mp3")
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")


def play_audio1(file):
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue


def terminal_codes(query: str) -> str:
    llm = ChatOllama(model="wizardcoder")
    response = llm([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ])
    return response.content


def generate_speech(text, output_file="output_audio.mp3"):
    API_KEY = "sk_4105d2b73b066c0c9009debe8d58acef8d6cea50b9a10dba"
    voice_id = "nPczCjzI2devNBz1zQrb"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75,
            "speed": 1.0,
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio saved as {output_file}")
        play_audio1(output_file)
    else:
        print("❌ Error:", response.status_code)
        print(response.text)


def coder(query: str) -> str:
    llm = ChatOllama(model="starcoder2:7b")
    response = llm([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ])
    return response.content


def extract_text_from_pdfs(pdf_paths: list[str]) -> str:
    all_text = ""

    for pdf_path in pdf_paths:
        try:
            # Try native text extraction first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        all_text += text + "\n"
                    else:
                        # Fallback to OCR for this page
                        image = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(image)
                        all_text += ocr_text + "\n"

        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")

    return all_text.strip()


def analyze_with_ollama(combined_input: str) -> str:
    # Expecting format: "Describe the image at /path/to/image"
    match = re.search(r"(.*?)(?:\s+at\s+|\s+)(\/.*)", combined_input)

    if match:
        query = match.group(1).strip()
        image_path = match.group(2).strip()
    else:
        return "Invalid input format. Use: 'Your query at /path/to/image'"

    prompt = f"{query} {image_path}"
    process = subprocess.Popen(
        ["ollama", "run", "llava:7b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=prompt)
    return stdout.strip()


def think_and_answer(query: str) -> str:
    llm = ChatOllama(model="deepseek-r1:7b")
    response = llm([HumanMessage(content=query)])
    return response.content


def deepsearch(query: str) -> str:
    try:
        response = master_agent_executor.invoke({"input": query, "agent_scratchpad": []})
        print("\n--- Master Agent Final Response ---\n")
        return response['output']
    except Exception as e:
        print(f"\nError during master agent execution: {e}")
        return "Check Ollama server, model names, API keys, and ReAct formatting compatibility."


def ssh_command_with_password(remote_command):
    username = "root"
    host = "192.168.1.2"
    password = "samrudha@8124"
    port = 8022
    command = [
        "sshpass", "-p", password,
        "ssh", f"{username}@{host}", "-p", str(port),
        remote_command
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
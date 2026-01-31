import os
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CLIENT_SECRET_FILE = "client_secret_190573199566-vj7hpkfqj91378i01j04drcudaj77hr7.apps.googleusercontent.com.json"
TOKEN_FILE = "token.json"
TIMEZONE = "Asia/Kolkata"


def authenticate_google():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh()
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def create_event(summary: str, start_time: str, end_time: str) -> str:
    service = authenticate_google()
    event = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": TIMEZONE},
        "end": {"dateTime": end_time, "timeZone": TIMEZONE},
    }
    created_event = service.events().insert(calendarId="primary", body=event).execute()
    return f"Event created: {created_event.get('htmlLink')}"


def list_upcoming_events(n: int = 5) -> str:
    service = authenticate_google()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    events_result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=n,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return "No upcoming events found."

    output = [f"{event['start'].get('dateTime', event['start'].get('date'))} — {event['summary']}" for event in events]
    return "\n".join(output)


def list_holidays(country: str = "india") -> str:
    service = authenticate_google()
    calendar_ids = {
        "india": "en.indian#holiday@group.v.calendar.google.com",
        "us": "en.usa#holiday@group.v.calendar.google.com",
        "uk": "en.uk#holiday@group.v.calendar.google.com",
    }
    calendar_id = calendar_ids.get(country.lower(), calendar_ids["india"])
    now = datetime.datetime.utcnow().isoformat() + "Z"

    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin=now,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return "No upcoming holidays found."

    output = [f"{event['start'].get('date')} — {event['summary']}" for event in events]
    return "\n".join(output)


def calendar_tool(query: str) -> str:
    try:
        query = query.lower()
        if "create event" in query:
            parts = query.split("from")
            title = parts[0].replace("create event", "").strip()
            times = parts[1].split("to")
            start_time = times[0].strip()
            end_time = times[1].strip()
            return create_event(title, start_time, end_time)

        elif "list events" in query or "show events" in query:
            return list_upcoming_events()

        elif "holidays" in query:
            return list_holidays()

        else:
            return "Invalid calendar query format."

    except Exception as e:
        return f"Error processing calendar query: {str(e)}"


def langchain_calendar_agent(user_input: str) -> str:
    llm = ChatOllama(model="jarvis2.0:latest")

    system_prompt = """
You are a smart assistant that processes calendar queries for a user.
Your task is to interpret the user's input and convert it into a simple command string such as:
- "Create event Meeting from 2025-05-13 10:00:00 to 2025-05-13 11:00:00"
- "List events"
- "List holidays"

Your output should ONLY be the interpreted command with no extra explanation.
"""

    chat = llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])

    interpreted_command = chat.content.strip()
    print(f"[LLM interpreted command] {interpreted_command}")
    return calendar_tool(interpreted_command)

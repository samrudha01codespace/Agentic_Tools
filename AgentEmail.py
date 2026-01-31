import os
import pickle
import base64
from typing import Any

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from email.mime.multipart import MIMEMultipart
from google.auth.transport.requests import Request
from email.mime.text import MIMEText
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain_ollama import ChatOllama
from oauthlib.uri_validate import query

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/gmail.modify',
          'https://www.googleapis.com/auth/gmail.compose',
          'https://www.googleapis.com/auth/gmail.send',
          'https://www.googleapis.com/auth/gmail.labels']

CREDENTIALS_FILE = 'client_secret.json'
ATTACHMENT_DIR = 'attachments/'


# Authenticate and create a service object for Gmail API
def authenticate_gmail_api():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service


# Function to list emails from Gmail
def list_messages(service, label_ids=['INBOX'], max_results=10, query=None):
    try:
        results = service.users().messages().list(userId='me', labelIds=label_ids, q=query,
                                                  maxResults=max_results).execute()
        messages = results.get('messages', [])
        return messages
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []


# Function to send an email
def send_message(service, to, subject, body):
    message = MIMEMultipart()
    message['to'] = to
    message['subject'] = subject
    msg = MIMEText(body)
    message.attach(msg)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    try:
        sent_message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        return f'Message sent successfully, ID: {sent_message["id"]}'
    except HttpError as error:
        return f'An error occurred: {error}'


# Download an attachment from an email
def download_attachment(service, msg_id, attachment_id, filename):
    try:
        attachment = service.users().messages().attachments().get(userId='me', messageId=msg_id,
                                                                  id=attachment_id).execute()
        data = attachment['body']['data']
        file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))

        if not os.path.exists(ATTACHMENT_DIR):
            os.makedirs(ATTACHMENT_DIR)

        file_path = os.path.join(ATTACHMENT_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)
        return f'Attachment downloaded to {file_path}'
    except HttpError as error:
        return f'An error occurred: {error}'


# Parse input
def parse_input(input_str, key):
    try:
        parts = [part.strip() for part in input_str.split(",")]
        for part in parts:
            if part.lower().startswith(f"{key}="):
                return part.split("=", 1)[1].strip()
    except Exception as e:
        return f"Error parsing input for {key}: {str(e)}"


# Define tools for LangChain
def create_gmail_tools(service):
    tools = [
        Tool(
            name="List Emails",
            func=lambda query=None: list_messages(service, query=query),
            description="List the most recent emails."
        ),
        Tool(
            name="Send Email",
            func=lambda input_str: send_message(
                service=service,
                to=parse_input(input_str, "to"),
                subject=parse_input(input_str, "subject"),
                body=parse_input(input_str, "body")
            ),
            description="Send an email. Provide input in the format: to=someone@example.com, subject=Your Subject, body=Message body"
        ),
        Tool(
            name="Download Attachment",
            func=lambda msg_id, attachment_id, filename: download_attachment(service, msg_id, attachment_id, filename),
            description="Download an attachment from a specific email."
        )
    ]
    return tools


# Initialize the LangChain agent with tools
def initialize_gmail_agent():
    service = authenticate_gmail_api()

    # Initialize LangChain LLM (using Ollama with Gemma as an example)
    llm = ChatOllama(model="jarvis2.0:latest", temperature=0.7)

    # Define a prompt template for the agent
    prompt = """
    You are a Gmail assistant. You can help manage Gmail operations like reading emails, sending emails, and managing attachments.
    Below are your commands:
    - "list emails": List the most recent emails in the inbox.
    - "send email to [email] with subject [subject] and body [body]": Send an email with the provided subject and body.
    - "download attachment from email [msg_id] with attachment ID [attachment_id]": Download an attachment from the specified email.

    User request: {input}
    """
    prompt_template = PromptTemplate(input_variables=["input"], template=prompt)

    # Create the LangChain chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Create tools for the agent
    tools = create_gmail_tools(service)

    # Initialize the LangChain agent
    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True
    )

    return agent, service


# Running the agent
def run_gmail_agent(query: str) -> Any | None:
    agent, service = initialize_gmail_agent()

    # Loop for continuous interaction with the user
    while True:
        user_input = input(query)
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Exiting...")
            break
        response = agent.run(user_input)
        print(response)
        return response

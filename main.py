import os
import threading
import time
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from twilio.rest import Client

from database.models import init_db
from tools.router import route_message

load_dotenv()

app = FastAPI(title="Morarc WhatsApp AI")

# Initialize Twilio Client
twilio_client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN"),
)


def split_message_for_whatsapp(text: str, max_len: int = 1500) -> List[str]:
    """Split long messages into safe WhatsApp-sized chunks, even for single huge blocks."""
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        return [""]

    chunks: List[str] = []
    current = ""

    for block in blocks:
        if len(block) > max_len:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            while start < len(block):
                end = min(start + max_len, len(block))
                if end < len(block):
                    split_at = block.rfind(" ", start, end)
                    if split_at <= start:
                        split_at = end
                else:
                    split_at = end

                part = block[start:split_at].strip()
                if part:
                    chunks.append(part)
                start = split_at
            continue

        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    return chunks


def process_and_send_reply(from_number: str, text: str) -> None:
    """
    Runs in the background. It asks the AI for a response and pushes
    it back to WhatsApp via Twilio, bypassing webhook timeout constraints.
    """
    try:
        ai_response = route_message(from_number, text)
        print(f"\n[OUTGOING] to {from_number}:\n{ai_response}\n")
    except Exception as exc:
        import traceback

        print(f"\n[CRITICAL ERROR] Thread crashed during route_message: {exc}")
        traceback.print_exc()
        return

    messages_to_send = split_message_for_whatsapp(ai_response, max_len=1500)

    try:
        for msg in messages_to_send:
            twilio_client.messages.create(
                body=msg,
                from_="whatsapp:+14155238886",
                to=from_number,
            )
            time.sleep(1)
    except Exception as exc:
        print(f"Error sending WhatsApp message: {exc}")


@app.on_event("startup")
def on_startup() -> None:
    print("Initializing Database...")
    init_db()
    print("Database Initialized.")


@app.get("/")
def health_check() -> dict:
    return {"status": "Morarc Backend is running."}


@app.post("/webhook")
async def twilio_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    print(f"\n[INCOMING] from {From}: {Body}")

    thread = threading.Thread(target=process_and_send_reply, args=(From, Body))
    thread.start()

    return PlainTextResponse(
        '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )

import os
import json
from database.models import init_db, SessionLocal, User, ConceptGraph
from tools.router import route_message
from core.memory import memory

# Inject test users into DB
db = SessionLocal()
personas = {
    "impatient_dev": "whatsapp:+11111111111",
    "deep_philosopher": "whatsapp:+22222222222",
    "confused_beginner": "whatsapp:+33333333333"
}

for name, phone in personas.items():
    if not db.query(User).filter(User.phone_number == phone).first():
        db.add(User(phone_number=phone, name=name))
    memory.clear_session(phone)
db.commit()
db.close()

def run_persona(name: str, phone: str, messages: list):
    print(f"\n=======================================================")
    print(f"🚀 RUNNING PERSONA: {name.upper()}")
    print(f"=======================================================")
    for i, msg in enumerate(messages):
        print(f"\n[{name}] Turn {i+1}: {msg}")
        try:
            resp = route_message(phone, msg)
            print(f"[MORARC AI]: {resp}")
        except Exception as e:
            print(f"[ERROR]: {e}")
            import traceback
            traceback.print_exc()
            break

# 1. The Impatient Developer (Wants code NOW, hates chatting)
impatient_chats = [
    "/articles how to build a FASTAPI server",
    "I literally just need the copy paste code and folder structure. Stop asking me questions.",
    "Give me the articles."
]

# 2. The Deep Philosopher (Wants profound subjective meaning, theoretical)
philosopher_chats = [
    "/articles the concept of determinism vs free will",
    "I lean towards hard determinism, specifically Sam Harris's view that free will is an illusion.",
    "I'm mostly interested in how this impacts our justice system. Because if no one chooses their actions, how can we punish them?",
    "Show me the articles on that."
]

# 3. The Confused Beginner (Doesn't know what they want, uses vague words)
beginner_chats = [
    "/articles I want to learn computers",
    "I don't know, like how to make websites or something.",
    "HTML I think? Someone mentioned HTML to me.",
    "Yeah how do I start with HTML?"
]

run_persona("Impatient Developer", personas["impatient_dev"], impatient_chats)
run_persona("Deep Philosopher", personas["deep_philosopher"], philosopher_chats)
run_persona("Confused Beginner", personas["confused_beginner"], beginner_chats)

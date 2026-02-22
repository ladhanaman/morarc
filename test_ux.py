"""
UX STRESS TEST — Morarc Conversational Audit
As a 20+ year UX Engineer, this script simulates three real users with distinct
emotional states and texting styles to surface friction in the AI's conversational flow.

Personas:
  1. The Skeptic       — concise, dismissive, tests "how good are you really?"
  2. The Wanderer      — vague, distracted, unsure what they want
  3. The Power User    — sharp, direct, already knows tools, gets into /articles fast
"""

import json
import time
from database.models import Base, engine, init_db, SessionLocal, User

# Fresh DB for each test run
Base.metadata.drop_all(bind=engine)
init_db()

from tools.router import route_message

def add_user(phone, name):
    db = SessionLocal()
    u = User(phone_number=phone, name=name)
    db.add(u)
    db.commit()
    db.close()

def run_persona(label, phone, name, turns):
    add_user(phone, name)
    print(f"\n{'='*60}")
    print(f"PERSONA: {label}")
    print(f"{'='*60}")
    for i, msg in enumerate(turns):
        print(f"\n[{label}] > {msg}")
        reply = route_message(phone, msg)
        print(f"\n[MORARC] >\n{reply}")
        print(f"\n--- UX NOTE: Message length = {len(reply)} chars ---")
        time.sleep(1) # Simulate natural pacing


# ─────────────────────────────────────────────────────────
# PERSONA 1: THE SKEPTIC
# Emotional State: Guarded, slightly challenging. Wants to
# see value before engaging. Classic "prove it" energy.
# UX Risk: AI over-explains and loses them immediately.
# ─────────────────────────────────────────────────────────
run_persona(
    label="THE SKEPTIC",
    phone="+10000000001",
    name="Skeptic",
    turns=[
        "hi",                                     # First message — triggers welcome
        "what even is this",                      # Vague dismissal — tests cliffhanger
        "okay so what can you actually do",       # Direct capability question
        "fine show me the articles thing",        # Surrenders — tests /articles entry
    ]
)

# ─────────────────────────────────────────────────────────
# PERSONA 2: THE WANDERER
# Emotional State: Curious but scattered. Texts in incomplete
# thoughts. Doesn't know what they want yet.
# UX Risk: AI asks too many questions at once and overwhelms.
# ─────────────────────────────────────────────────────────
run_persona(
    label="THE WANDERER",
    phone="+10000000002",
    name="Wanderer",
    turns=[
        "hey so i've been thinking about like consciousness",  # First message — welcome
        "yeah like is it an illusion or real",                 # Vague follow-up
        "i read something about the hard problem",             # Drops a concept
        "can you find me stuff to read on that",               # Implicit /articles ask
    ]
)

# ─────────────────────────────────────────────────────────
# PERSONA 3: THE POWER USER
# Emotional State: Efficient, direct, zero patience for fluff.
# Already knows what they want. Jumps straight into the tool.
# UX Risk: AI is still chatty/welcoming when user wants results.
# ─────────────────────────────────────────────────────────
run_persona(
    label="THE POWER USER",
    phone="+10000000003",
    name="Power User",
    turns=[
        "/articles the stoic philosophy of Marcus Aurelius",   # First message — triggers welcome + tool
        "specifically on his view of suffering",               # Narrows intent
        "academic papers preferably",                          # Specifies archetype
    ]
)

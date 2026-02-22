from core.llm import generate_completion
from core.memory import Session, memory
from database.models import SessionLocal, User
from tools.articles import handle_articles_tool
from tools.invite import handle_invite_tool

MASTER_NUMBER = "whatsapp:+917340068665"


def route_message(phone_number: str, message: str) -> str:
    """
    The main routing engine for Morarc.
    Handles session memory, stacked tools, and general chat.
    """
    with memory.get_session_lock(phone_number):
        message = message.strip()

        is_authorized = False
        is_master = phone_number == MASTER_NUMBER
        needs_welcome = False

        db = SessionLocal()
        try:
            if is_master:
                is_authorized = True
            else:
                user = db.query(User).filter(User.phone_number == phone_number).first()
                if user:
                    is_authorized = True
                    if not user.has_been_welcomed:
                        needs_welcome = True
                        user.has_been_welcomed = True
                        db.commit()
        finally:
            db.close()

        if not is_authorized:
            return "Unauthorized. You are not registered to interact with Morarc."

        session = memory.get_or_create_session(phone_number)

        welcome_prefix = ""
        if needs_welcome:
            welcome_prefix = (
                "Morarc.\n\n"
                "Tools:\n"
                "- articles — deep semantic web search, 3 curated results\n\n"
                "To use a tool, prefix it with /\n"
                "Example: /articles quantum computing\n\n"
                "Or just talk.\n\n"
                "---\n"
            )

        if message.startswith("/stop"):
            memory.clear_session(phone_number)
            return welcome_prefix + "Session terminated and memory cleared. Start fresh anytime."

        if message.startswith("/"):
            parts = message.split(maxsplit=1)
            tool_name = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if tool_name == "/invite":
                if not is_master:
                    return welcome_prefix + "Error: Only the admin can use the /invite command."
                return welcome_prefix + handle_invite_tool(args)

            if tool_name == "/articles":
                current_tool = session.get_current_tool()
                if current_tool:
                    if current_tool == "/articles":
                        if args:
                            return welcome_prefix + route_tool(session, args)
                        return (
                            welcome_prefix
                            + "You are already in /articles. Share your topic, or send 'done' to exit."
                        )
                    return welcome_prefix + f"Finish or stop '{current_tool}' before starting /articles."

                session.push_tool(tool_name)
                if args:
                    memory.retrieve_and_inject_rag(session, args)
                    return welcome_prefix + route_tool(session, args)
                return welcome_prefix + "What concept would you like to explore?"

            return welcome_prefix + f"Unknown tool '{tool_name}'. Currently supported: /articles"

        articles_triggers = [
            "show me articles",
            "show me the articles",
            "try the articles",
            "use articles",
            "launch articles",
            "start articles",
            "find me articles",
        ]
        if any(trigger in message.lower() for trigger in articles_triggers):
            current_tool = session.get_current_tool()
            if current_tool:
                if current_tool == "/articles":
                    return (
                        welcome_prefix
                        + "You are already in /articles. Share your topic, or send 'done' to exit."
                    )
                return welcome_prefix + f"Finish or stop '{current_tool}' before starting /articles."
            session.push_tool("/articles")
            return welcome_prefix + "What concept would you like to explore?"

        return welcome_prefix + route_tool(session, message)


def route_tool(session: Session, message: str) -> str:
    current_tool = session.get_current_tool()

    if not current_tool:
        session.add_message("user", message)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Morarc, an 'Intellectual Sparring Partner'. Sharp, highly intelligent, slightly provocative. "
                    "RULE 1 - PING PONG: One action per message. If user asks a question -> only answer it. Full stop. "
                    "If user makes a statement -> only ask one sharp question back. Never do both in the same message. "
                    "RULE 2 - CLIFFHANGER: Never dump. If explaining a complex topic, give the single most interesting sentence, then stop. Let the user pull more from you. "
                    "RULE 3 - NO FILLER: Never say 'Great question!', 'Certainly!', 'I'd be happy to help!', or any other filler phrases. Start with the substance. "
                    "RULE 4 - TOOLS: If asked what you can do, say exactly this: 'I have /articles <topic>. It does deep semantic web scrapes and finds you 3 precise articles. Try it.' Nothing more. "
                    "RULE 5 - NO EMOJIS: Forbidden."
                ),
            }
        ] + session.chat_history

        response = generate_completion(messages)
        session.add_message("assistant", response)
        return response

    if current_tool == "/articles":
        return handle_articles_tool(session, message)

    return "Error: Unexpected tool state."

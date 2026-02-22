from database.models import SessionLocal, User

def handle_invite_tool(args: str) -> str:
    """
    Format: /invite +919999999999 John Doe
    Admin only tool to add authorized users to the SQLite/Turso database.
    """
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return "Usage: /invite <phone_number> <name>"
    
    phone = parts[0].strip()
    name = parts[1].strip()
    
    # Twilio Sandbox format ensures numbers prefixed with whatsapp:
    if not phone.startswith("whatsapp:"):
        # Auto-correct common mistakes, ensure it starts with +
        if not phone.startswith("+"):
            phone = f"+{phone}"
        phone = f"whatsapp:{phone}"
        
    db = SessionLocal()
    try:
        # Check if already exists
        user = db.query(User).filter(User.phone_number == phone).first()
        if user:
            return f"User {name} ({phone}) is already registered and authorized."
        
        new_user = User(phone_number=phone, name=name)
        db.add(new_user)
        db.commit()
        return f"Successfully invited {name} ({phone}) to Morarc. They can now send messages to this number."
    except Exception as e:
        db.rollback()
        return f"Database error occurred: {str(e)}"
    finally:
        db.close()

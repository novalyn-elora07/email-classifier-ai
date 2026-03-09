urgency_keywords = {
    "high": ["urgent", "asap", "immediately", "not working", "critical"],
    "medium": ["soon", "important", "please respond"],
    "low": ["whenever", "no rush", "later"]
}

def keyword_urgency(text):

    text = text.lower()

    for word in urgency_keywords["high"]:
        if word in text:
            return "high"

    for word in urgency_keywords["medium"]:
        if word in text:
            return "medium"

    return "low"

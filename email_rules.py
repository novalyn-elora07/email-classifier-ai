def urgency_score(email_text):
    email_text = email_text.lower()

    score = 1

    urgent_keywords = ["urgent", "asap", "immediately", "important", "help"]

    for word in urgent_keywords:
        if word in email_text:
            score += 2

    if score > 10:
        score = 10

    return score
def rule_classifier(email_text):
    email_text = email_text.lower()

    if "not working" in email_text:
        return "complaint"
    
    elif "love" in email_text:
        return "feedback"
    
    elif "buy now" in email_text:
        return "spam"
    
    else:
        return "general"
   # sample email for testing
email = "My login is not working please fix this asap"

print("Urgency Score:", urgency_score(email))
print("Category:", rule_classifier(email))
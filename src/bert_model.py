from transformers import pipeline

classifier = pipeline("text-classification")

text = "My account is not working"

result = classifier(text)

print(result)
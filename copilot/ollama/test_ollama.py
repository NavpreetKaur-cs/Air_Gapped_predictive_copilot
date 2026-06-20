
from ollama import chat

response = chat(
    model="llama3:8b",
    messages=[
        {
            "role": "user",
            "content": "What is MPLS?"
        }
    ]
)

print("\nResponse:\n")
print(response["message"]["content"])
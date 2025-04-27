# import openai

# openai.api_key ="your key"


# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hello!"}]
# )

# print(response.choices[0].message['content'])

from openai import OpenAI

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key="f39db8e8c9f0429dbd05276e2263b51f",    
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a one-sentence story about numbers."}]
)

print(response.choices[0].message.content)
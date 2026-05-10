from openai import OpenAI

client = OpenAI(
    api_key="")

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Write a on-line joke about programmeers "
)

print(response.output_text)

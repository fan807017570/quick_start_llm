from openai import OpenAI

client = OpenAI(
    api_key="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")

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Write a on-line joke about programmeers "
)

print(response.output_text)

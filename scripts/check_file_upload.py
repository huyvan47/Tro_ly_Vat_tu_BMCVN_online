from openai import OpenAI
client = OpenAI(api_key="sk-................")

files = client.files.list()
for f in files.data:
    print(f.id, f.filename, f.purpose)

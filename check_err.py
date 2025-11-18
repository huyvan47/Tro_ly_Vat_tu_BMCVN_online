from openai import OpenAI
client = OpenAI(api_key="sk-........")

job_id = "ftjob-U8yPmWlxkdr45uVPxTMFB3zR"
print(client.fine_tuning.jobs.retrieve(job_id))
for e in client.fine_tuning.jobs.list_events(job_id=job_id, limit=200):
    print(e)
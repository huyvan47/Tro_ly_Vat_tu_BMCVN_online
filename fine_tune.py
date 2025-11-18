from openai import OpenAI
client = OpenAI(api_key="sk-proj-................")

train = client.files.create(file=open("vattuphu-train.jsonl", "rb"), purpose="fine-tune")
valid = client.files.create(file=open("vattuphu-valid.jsonl", "rb"), purpose="fine-tune")

# create job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini-2024-07-18",
    training_file=train.id,
    validation_file=valid.id,
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 4,
        "learning_rate_multiplier": 1.0
    }
)

print("Job:", job.id)

# 3) Chờ hoàn tất (đơn giản hoá)
while True:
    j = client.fine_tuning.jobs.retrieve(job.id)
    if j.status in ["succeeded","failed","cancelled"]:
        print(j.status, getattr(j, "fine_tuned_model", None))
        break

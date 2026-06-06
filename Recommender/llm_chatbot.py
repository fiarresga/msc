import pandas as pd
import openai
import time
import warnings

warnings.filterwarnings("ignore")


with open("openai_key.txt", "r") as f:
    api_key = f.read().strip()

client = openai.OpenAI(api_key=api_key)


met_df = pd.read_csv("met_values.csv")
cpa_df = pd.read_csv("cpa.csv")


assistant = client.beta.assistants.create(
    name="Exercise Advisor",
    model="gpt-4",
    tools=[],
    instructions=(
        "You are a helpful assistant that helps users choose physical activities "
        "based on their MET range and personal preferences. You will be given a MET range "
        "and a list of activities in a markdown table. The user will also describe their goals, "
        "preferences, or limitations (e.g., endurance, outdoor, knee pain). Recommend the most suitable "
        "activities from the table and explain why. Use markdown formatting when helpful."
    )
)


thread = client.beta.threads.create()


def get_met_range(user_id):
    row = met_df[met_df['eid'] == int(user_id)]
    if row.empty:
        raise ValueError("User ID not found.")
    return int(row['met_min'].values[0]), int(row['met_max'].values[0])


def filter_activities(met_min, met_max):
    return cpa_df[(cpa_df['MET'] >= met_min) & (cpa_df['MET'] <= met_max)]


def send_message_and_get_reply(message):
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=[{"type": "text", "text": message}]
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            raise RuntimeError("Run failed.")
        time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread.id, limit=1)
    latest_message = messages.data[0]
    return latest_message.content[0].text.value if latest_message.role == "assistant" else "No assistant response."



initial_prompt = (
    "Hello! I’d like to find exercises. Please ask me for my user ID and I’ll provide it next."
)
print("Assistant:", send_message_and_get_reply(initial_prompt))

id_provided = False
conversation_context = ""

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break

    if not id_provided:
        try:
            user_id = str(user_input)
            met_min, met_max = get_met_range(user_id)
            id_provided = True

            filtered_df = filter_activities(met_min, met_max)
            activity_sample = filtered_df.head(700).to_markdown(index=False)

            context_intro = (
                f"The user ID is {user_id}, which maps to a MET range of {met_min}–{met_max}.\n\n"
                f"Here are 700 activities from my dataset within this MET range:\n{activity_sample}\n\n"
                f"Please ask the user relevant questions to help personalize your suggestions."
            )

            print("Assistant:", send_message_and_get_reply(context_intro))

        except Exception as e:
            print("Error:", e)

    else:
        user_preferences = user_input.strip()

        message = (
            f"The user previously provided the ID {user_id}, which maps to a MET range of {met_min}–{met_max}.\n\n"
            f"Here are 700 activities from my dataset within this range:\n{activity_sample}\n\n"
            f"The user now said: \"{user_preferences}\"\n\n"
            f"Based on this, please recommend suitable activities and explain why. "
            f"Consider any goals, preferences (e.g. outdoor, social), or limitations (e.g. knee problems) they mention."
            f"When printing activities from the list, offer an explanation in simple terms and interpret the activity in simple terms for the user to understand."
        )

        print("Assistant:", send_message_and_get_reply(message))

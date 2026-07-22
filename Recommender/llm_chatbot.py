import time
import warnings
from pathlib import Path

import pandas as pd
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL = "gpt-5-nano"
MAX_ACTIVITIES = 100

SYSTEM_INSTRUCTIONS = """
You are a helpful exercise advisor.

You help users choose physical activities based on:
- Their allowed MET range
- Their goals
- Their preferences
- Their physical limitations
- The activities provided from the user's dataset

Only recommend activities that appear in the provided dataset.

When making recommendations:
1. Use simple, accessible language.
2. Explain what each activity involves.
3. Explain why it matches the user's goals and preferences.
4. Respect limitations such as knee pain, reduced mobility, or low fitness.
5. Do not diagnose medical conditions.
6. When a limitation could require professional medical advice, say so clearly.
7. Use concise Markdown formatting when helpful.
"""


# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------

api_key_path = Path("openai_key.txt")

if not api_key_path.exists():
    raise FileNotFoundError(
        "Could not find openai_key.txt. Create the file and place your "
        "OpenAI API key inside it."
    )

api_key = api_key_path.read_text(encoding="utf-8").strip()

if not api_key:
    raise ValueError("openai_key.txt is empty.")

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------

met_df = pd.read_csv("met_values.csv")
cpa_df = pd.read_csv("cpa.csv")


required_met_columns = {"eid", "met_min", "met_max"}
required_cpa_columns = {"MET"}

missing_met_columns = required_met_columns - set(met_df.columns)
missing_cpa_columns = required_cpa_columns - set(cpa_df.columns)

if missing_met_columns:
    raise ValueError(
        f"met_values.csv is missing columns: {sorted(missing_met_columns)}"
    )

if missing_cpa_columns:
    raise ValueError(
        f"cpa.csv is missing columns: {sorted(missing_cpa_columns)}"
    )


# Convert relevant columns once, rather than during every search.
met_df["eid"] = pd.to_numeric(met_df["eid"], errors="coerce")
met_df["met_min"] = pd.to_numeric(met_df["met_min"], errors="coerce")
met_df["met_max"] = pd.to_numeric(met_df["met_max"], errors="coerce")
cpa_df["MET"] = pd.to_numeric(cpa_df["MET"], errors="coerce")


# ---------------------------------------------------------------------
# Dataset functions
# ---------------------------------------------------------------------

def get_met_range(user_id: str) -> tuple[float, float]:
    """Return the MET range associated with a user ID."""

    try:
        numeric_user_id = int(user_id)
    except ValueError as exc:
        raise ValueError("The user ID must be a number.") from exc

    row = met_df.loc[met_df["eid"] == numeric_user_id]

    if row.empty:
        raise ValueError(f"User ID {numeric_user_id} was not found.")

    met_min = row.iloc[0]["met_min"]
    met_max = row.iloc[0]["met_max"]

    if pd.isna(met_min) or pd.isna(met_max):
        raise ValueError(
            f"User ID {numeric_user_id} does not have a valid MET range."
        )

    if met_min > met_max:
        raise ValueError(
            f"Invalid MET range for user {numeric_user_id}: "
            f"{met_min}-{met_max}."
        )

    return float(met_min), float(met_max)


def filter_activities(
    met_min: float,
    met_max: float
) -> pd.DataFrame:
    """Return activities inside the specified MET range."""

    filtered = cpa_df.loc[
        cpa_df["MET"].between(met_min, met_max, inclusive="both")
    ].copy()

    return filtered.sort_values("MET").reset_index(drop=True)


def dataframe_to_compact_text(
    dataframe: pd.DataFrame,
    max_rows: int = MAX_ACTIVITIES
) -> str:
    """
    Convert activities into compact CSV-like text.

    This is normally more token-efficient than a Markdown table.
    """

    sample = dataframe.head(max_rows)

    return sample.to_csv(
        index=False,
        lineterminator="\n"
    )


# ---------------------------------------------------------------------
# OpenAI API function
# ---------------------------------------------------------------------

def get_model_reply(
    message: str,
    previous_response_id: str | None = None,
    max_retries: int = 3
) -> tuple[str, str]:
    """
    Send a message through the Responses API.

    Returns:
        A tuple containing:
        - response text
        - response ID for continuing the conversation
    """

    for attempt in range(max_retries):
        try:
            request = {
                "model": MODEL,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": message,
                "reasoning": {
                    "effort": "minimal"
                },
                "max_output_tokens": 2000,
            }

            if previous_response_id is not None:
                request["previous_response_id"] = previous_response_id

            response = client.responses.create(**request)

            print(response.model_dump_json(indent=2))

            reply = ""

            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            reply += content.text

            reply = reply.strip()

            if not reply:
                reply = "I could not generate a response."

            return reply, response.id

        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            delay = 2 ** attempt
            print(f"Rate limit reached. Retrying in {delay} seconds...")
            time.sleep(delay)

        except APIConnectionError as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    "Could not connect to the OpenAI API."
                ) from exc

            delay = 2 ** attempt
            print(f"Connection problem. Retrying in {delay} seconds...")
            time.sleep(delay)

        except APIStatusError as exc:
            raise RuntimeError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

    raise RuntimeError("The request failed after multiple attempts.")


# ---------------------------------------------------------------------
# Main conversation
# ---------------------------------------------------------------------

def main() -> None:
    print(
        "Assistant: Hello! I can help you find suitable physical "
        "activities. Please enter your user ID."
    )

    user_id = None
    met_min = None
    met_max = None
    previous_response_id = None

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Assistant: Please enter a response.")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        # First valid input should be the user ID.
        if user_id is None:
            try:
                met_min, met_max = get_met_range(user_input)
                filtered_df = filter_activities(met_min, met_max)

                if filtered_df.empty:
                    print(
                        "Assistant: I could not find any activities "
                        f"between {met_min:g} and {met_max:g} METs."
                    )
                    continue

                user_id = user_input
                activity_count = min(len(filtered_df), MAX_ACTIVITIES)
                activities_text = dataframe_to_compact_text(filtered_df)

                dataset_message = f"""
The user's ID is {user_input}.

Their permitted MET range is:
- Minimum MET: {met_min:.1f}
- Maximum MET: {met_max:.1f}

The following dataset contains {activity_count} activities.

{activities_text}

Only recommend activities contained in this dataset.

Ask the user one or two useful questions about their goals,
preferences, fitness level, or physical limitations before making
recommendations.
"""

                try:
                    reply, previous_response_id = get_model_reply(
                        dataset_message
                    )
                except RuntimeError as exc:
                    print(f"Assistant: Something went wrong: {exc}")
                    user_id = None
                    continue

                print(f"Assistant: {reply}")

            except ValueError as exc:
                print(f"Assistant: {exc} Please try again.")
                continue

        # Subsequent turns: user_id is already set, so just relay the
        # user's message through the existing conversation.
        else:
            try:
                reply, previous_response_id = get_model_reply(
                    user_input,
                    previous_response_id=previous_response_id
                )
            except RuntimeError as exc:
                print(f"Assistant: Something went wrong: {exc}")
                continue

            print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()

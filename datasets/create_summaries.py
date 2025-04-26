import pandas as pd
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

api_key =  os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

# === Prompts ===
PROMPTS = {
    "traditional": "Write a short summary describing the ongoing conversation. Keep the summary under 80 words.",
    "scd": "Write a short summary capturing the trajectory of the ongoing conversation. Do not include specific topics, claims, or arguments from the conversation. Instead, try to capture how the speakers' sentiments, intentions, and conversational/persuasive strategies change or persist throughout the conversation. Limit the trajectory summary to 80 words.",
    "relational": "At this point in the ongoing conversation, infer how each participant feels about the other. Focus on their apparent level of respect, trust, frustration, alignment, or rapport. Do not refer to specific content. Limit the description to 80 words.",
    "scm": "Using the Stereotype Content Model (SCM) as your framework, assess the ongoing conversation by quantifying each participant's warmth and competence. For each participant, categorize their perceived warmth and competence as low or high. Use the following format for your output:\n\nuser1:\n  warmth: <low/high>\n  competence: <low/high>\n  explanation: <concise explanation within 80 words>\n\nuser2:\n  warmth: <low/high>\n  competence: <low/high>\n  explanation: <concise explanation within 80 words>\n\nAvoid referencing specific topics or arguments.",
    "politeness_theory_stage1": """Using the politeness theory of Brown and Levinson, summarize the ongoing conversation by describing the overall use of face management strategies. Face, conceptualized as an individual's positive claim to social value in social interactions, was introduced by Erving Goffman through his theories of "face" and "facework." Brown and Levinson built on this by defining two universal aspects of face: positive face—the desire to be liked, admired, and approved of—and negative face—the desire for autonomy, freedom from imposition, and personal space. A face-threatening act (FTA) occurs when an utterance or behavior risks damaging the speaker's or hearer's face. A negative face-threatening act obstructs the hearer's freedom of choice or imposes on them, while a positive face-threatening act signals disregard for the hearer's self-image, feelings, or desire to be appreciated. Despite their inevitability in interaction, these acts can be softened using politeness strategies. Conversely, a face-raising act (FRA) strengthens or affirms the hearer's face—either by showing admiration, agreement, or granting autonomy—thus reinforcing social value and rapport. In this context, examine how the conversation reflects the use of FTAs and FRAs, and how these shaped the evolving interpersonal dynamics. Provide a detailed explanation that explains how participants' positive and negative face concerns were addressed throughout the interaction, highlighting the influence of these politeness strategies on interpersonal dynamics. Avoid referencing specific topics or arguments."""
}

SHORTEN_PROMPT = (
    "Rewrite the following explanation into a more concise summary, limited to 80 words. "
    "Preserve the key insights about how face-threatening and face-raising acts were managed, "
    "as well as the influence of politeness strategies on interpersonal dynamics. "
    "Avoid repeating definitions or unnecessary detail, and do not reference specific conversational content.\n\n"
)

def generate_summary(dialogue, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for analyzing conversations."},
                {"role": "user", "content": f"{prompt}\n\nConversation:\n{dialogue}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"

def shorten_summary(long_summary):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that shortens explanations."},
                {"role": "user", "content": SHORTEN_PROMPT + long_summary}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"

def format_conversation(group):
    return "\n".join(f"{row['speaker']}: {row['utterance']}" for _, row in group.iterrows())

def main():
    parser = argparse.ArgumentParser(description="Generate multi-type summaries for dialogue data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name (e.g., p4g)")
    parser.add_argument("--ratio", type=str, required=True, help="Truncation ratio (e.g., 0.5)")
    parser.add_argument("--summaries", nargs="+", choices=list(PROMPTS.keys()) + ["politeness_theory_stage2"],
                        required=True, help="Summary types to generate")
    args = parser.parse_args()

    # File paths
    input_file = f"datasets/{args.dataset}/final/ratio_{args.ratio}.csv"
    output_file = f"datasets/{args.dataset}/final/ratio_{args.ratio}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.read_csv(input_file)

    # Build dialogue blocks
    dialogue_blocks = {}
    for dialogue_id, group in df.groupby("dialogue_id"):
        dialogue_blocks[dialogue_id] = format_conversation(group)

    # DataFrame to hold dialogue-level summaries
    dialogue_summaries = pd.DataFrame({"dialogue_id": list(dialogue_blocks.keys())})
    
    for summary_type in args.summaries:
        col = f"{summary_type}_summary"
        print(f"\nGenerating: {summary_type}")

        if summary_type == "politeness_theory_stage2":
            if "politeness_theory_stage1_summary" not in dialogue_summaries.columns:
                raise ValueError("Stage 2 requires stage 1 to be generated first.")

            dialogue_summaries[col] = [
                shorten_summary(row["politeness_theory_stage1_summary"])
                if pd.notna(row["politeness_theory_stage1_summary"]) else ""
                for _, row in tqdm(dialogue_summaries.iterrows(), total=len(dialogue_summaries))
            ]
        else:
            prompt = PROMPTS[summary_type]
            dialogue_summaries[col] = [
                generate_summary(dialogue_blocks[did], prompt)
                for did in tqdm(dialogue_summaries["dialogue_id"])
            ]

    # Merge back into utterance-level DataFrame
    df_final = df.merge(dialogue_summaries, on="dialogue_id", how="left")

    # Save final output
    df_final.to_csv(output_file, index=False)
    print(f"Saved summarized data to: {output_file}")

if __name__ == "__main__":
    main()

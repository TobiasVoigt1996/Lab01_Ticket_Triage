import pandas as pd
from transformers import pipeline

# --------------------------------------------------
# Settings
# --------------------------------------------------

DATASET_PATH = "ticket_dataset.csv"
NUMBER_OF_TICKETS = 25

CATEGORY_LABELS = [
    "technical problem",
    "billing issue",
    "delivery issue",
    "account access issue",
    "security incident",
    "contract cancellation",
    "general request"
]

PRIORITY_LABELS = [
    "high priority",
    "medium priority",
    "low priority"
]

# --------------------------------------------------
# Dataset
# --------------------------------------------------

print("Load Data...")

df = pd.read_csv(DATASET_PATH)

# Nur englische Tickets nutzen, damit die Modelle stabiler funktionieren
df = df[df["language"] == "en"].copy()

# Leere Werte entfernen
df = df.dropna(subset=["subject", "body"])

# Für kurze Laufzeit nur einige Tickets
df = df.head(NUMBER_OF_TICKETS)

print(f"Geladene Tickets für Analyse: {len(df)}")

# --------------------------------------------------
# Model
# --------------------------------------------------

print("Load Hugging-Face-Model...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

summarizer = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Models loaded.\n")

# --------------------------------------------------
# Ticketanalysis
# --------------------------------------------------

results = []

for index, row in df.iterrows():
    subject = str(row["subject"])
    body = str(row["body"])

    ticket_text = subject + ". " + body

    # Texte kürzen, damit das Modell stabil bleibt
    ticket_text_short = ticket_text[:1500]

    print(f"Analysiere Ticket {index}...")

    # Schritt A: Kategorie erkennen
    category_result = classifier(ticket_text_short, CATEGORY_LABELS)
    predicted_category = category_result["labels"][0]
    category_score = round(category_result["scores"][0], 3)

    # Schritt B: Priorität bestimmen
    # Wichtig: Ergebnis aus Schritt A wird hier als Input genutzt.
    priority_input = (
        f"Predicted ticket category: {predicted_category}. "
        f"Ticket text: {ticket_text_short}"
    )

    priority_result = classifier(priority_input, PRIORITY_LABELS)
    predicted_priority = priority_result["labels"][0]
    priority_score = round(priority_result["scores"][0], 3)

    # Schritt C: Zusammenfassung
    summary_input = ticket_text_short[:1000]

    summary_prompt = "Support ticket summary: " + summary_input[:600]

    generated = summarizer(
    summary_prompt,
    max_length=90,
    do_sample=False,
    truncation=True
    )[0]["generated_text"]

    summary = generated.replace(summary_prompt, "").strip()

    results.append({
        "subject": subject,
        "predicted_category": predicted_category,
        "category_score": category_score,
        "expected_type": row.get("type", ""),
        "predicted_priority": predicted_priority,
        "priority_score": priority_score,
        "expected_priority": row.get("priority", ""),
        "summary": summary
    })

# --------------------------------------------------
# Safe Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nResults:")
print(results_df[[
    "subject",
    "predicted_category",
    "expected_type",
    "predicted_priority",
    "expected_priority",
    "summary"
]].to_string(index=False))

results_df.to_csv("ticket_results.csv", index=False, encoding="utf-8")

print("\nDone. Results saved in ticket_results.csv.")
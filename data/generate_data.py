"""
Generates a small synthetic financial-news sentiment dataset for demo
purposes. Swap this out for a real dataset such as `financial_phrasebank`
or `twitter-financial-news-sentiment` (both on Hugging Face Datasets) for
production-grade training.
"""
import json
import random

random.seed(42)

positive_templates = [
    "{company} reported record quarterly profits, beating analyst expectations.",
    "{company} shares surged after strong earnings guidance for next year.",
    "Analysts upgraded {company} stock citing robust revenue growth.",
    "{company} announced a major expansion into new international markets.",
    "{company} posted better-than-expected margins driven by cost efficiency.",
]

negative_templates = [
    "{company} shares plunged after missing revenue targets significantly.",
    "{company} announced layoffs amid declining sales and rising debt.",
    "Analysts downgraded {company} citing weak cash flow and shrinking margins.",
    "{company} faces regulatory investigation over accounting irregularities.",
    "{company} warned of a sharp profit decline for the upcoming quarter.",
]

neutral_templates = [
    "{company} will release its quarterly earnings report next Tuesday.",
    "{company} appointed a new board member effective immediately.",
    "{company} maintained its market share, matching analyst expectations.",
    "{company} held its annual shareholder meeting this week.",
    "{company} completed a routine refinancing of existing debt.",
]

companies = [
    "Apex Financial", "Meridian Bank", "NovaTech Holdings", "Sterling Capital",
    "BlueRiver Corp", "Horizon Industries", "Pinnacle Group", "Orion Markets",
    "Falcon Enterprises", "Quantum Partners",
]

records = []
for label, templates in [("positive", positive_templates), ("negative", negative_templates), ("neutral", neutral_templates)]:
    for template in templates:
        for company in companies:
            records.append({"text": template.format(company=company), "label": label})

random.shuffle(records)

split = int(len(records) * 0.8)
train, test = records[:split], records[split:]

with open("data/train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")

with open("data/test.jsonl", "w") as f:
    for r in test:
        f.write(json.dumps(r) + "\n")

print(f"Train: {len(train)} | Test: {len(test)}")

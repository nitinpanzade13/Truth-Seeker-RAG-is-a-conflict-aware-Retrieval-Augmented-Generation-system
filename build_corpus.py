import wikipediaapi
import os

# Create corpus folder
os.makedirs("corpus", exist_ok=True)

# Define technology + finance topics
topics = [
    "Apple Inc",
    "Microsoft",
    "Google",
    "Amazon (company)",
    "Meta Platforms",
    "Tesla, Inc.",
    "Nvidia",
    "Stock market",
    "Market capitalization",
    "Financial crisis",
    "Inflation",
    "Recession",
    "NASDAQ",
    "S&P 500",
    "Dow Jones Industrial Average",
    "Venture capital",
    "Artificial intelligence",
    "Big Tech",
    "Cloud computing",
    "E-commerce"
]

wiki = wikipediaapi.Wikipedia('TruthSeekerRAG/1.0', 'en')

for topic in topics:
    page = wiki.page(topic)

    if page.exists():
        filename = topic.replace(" ", "_").replace(",", "") + ".txt"
        filepath = os.path.join("corpus", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page.text)

        print(f"Saved: {filename}")
    else:
        print(f"Skipped: {topic}")
import os
from groq import Groq


# Load API key securely from environment variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Set it using environment variables.")

client = Groq(api_key=api_key)


def generate_answer(query, ranked_docs):
    """
    Generates final answer using Groq Llama-3 model.
    """

    # Build context from top ranked documents
    context = "\n\n".join(
        [f"Source {i+1} (Confidence: {round(score,3)}):\n{doc}"
         for i, (doc, score) in enumerate(ranked_docs)]
    )

    prompt = f"""
You are Truth-Seeker RAG, an evidence-aware AI system.

User Question:
{query}

Retrieved Evidence:
{context}

Instructions:
1. Use the highest-confidence evidence first.
2. If evidence conflicts, explicitly explain the contradiction.
3. If evidence agrees, state that no major conflict was detected.
4. Provide a clear, factual, and concise answer.
5. Do NOT hallucinate information not present in the evidence.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response from Groq: {str(e)}"
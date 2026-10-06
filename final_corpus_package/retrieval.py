"""
Final retrieval module for RomanAgent-Reliability -- TF-IDF with stemming
over the reviewed 6-document, 26-chunk policy corpus.

Change log vs the earlier version: added Porter stemming, because plain
TF-IDF without it failed a real query during review ('gym membership
expense' matched the wrong chunk since 'membership' vs 'memberships' are
different tokens with no stemming). Worth citing as a known limitation of
plain TF-IDF in your methodology section even with the fix -- stemming
helps English morphology, but won't help nearly as much once Roman Urdu
queries are involved, since Porter stemming is English-specific. That gap
is itself a legitimate thing your benchmark should surface.
"""
from config import GEMINI_API_KEY
import json
import os
import re
from google import genai
from google.genai import types



roman_urdu={
    "q1":("kitni chutiyan kr skta hon")
}

client=genai.Client(api_key=GEMINI_API_KEY)

def gen_embeddings(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=1536
        )
    )

    return response.embeddings[0].values



#from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

scores=[]

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "final_corpus_chunks.json")

# Minimal manual stopword list -- avoids sklearn's built-in list (which
# isn't stemmed and triggers a mismatch warning against a stemmed tokenizer).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for",
    "with", "on", "at", "by", "be", "this", "that", "it", "as", "from",
}

#_stemmer = PorterStemmer()
_vectorizer = None
_matrix = None
_chunks: list[dict] = []


def _stem_tokenizer(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
   # return [_stemmer.stem(t) for t in tokens if t not in _STOPWORDS]


def _load_index() -> None:
    global _vectorizer, _matrix, _chunks
    if _vectorizer is not None:
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    texts = [c["text"] for c in _chunks]
    _vectorizer = TfidfVectorizer(tokenizer=_stem_tokenizer, token_pattern=None)
    _matrix = _vectorizer.fit_transform(texts)

with open("final_corpus_package/final_corpus_chunks.json","r",encoding="utf-8") as f:
    corpus=json.load(f)
    print("Total chunks:", len(corpus))
    print("First chunk:", corpus[0]["chunk_id"])
    print("Last chunk:", corpus[-1]["chunk_id"])

def search_chunks(querry:str,top_k: int = 5)->list[dict]:
    input_embeddings=gen_embeddings(querry)
    for chunks in corpus:
        score=cosine_similarity([input_embeddings],[chunks["embedding"]])[0][0]

        scores.append({
            "chunk_id":chunks["chunk_id"],
            "text":chunks["text"],
            "score":float(score)
        })
    scores.sort(
            key=lambda x:x["score"],
            reverse=True
        )
    print("\nQUERY:", querry)

    print("\nTOP 5 RESULTS:")
    for result in scores[:5]:
        print(
            result["chunk_id"],
            "->",
            round(result["score"], 4)
        )
    return scores[:top_k]

def generate_rag_prompt(scores:list[dict],querry:str):
    if not scores:
        context="(No relevant policy was found in the corpus)"
    else:
        context="\n\n".join(f"[{c["chunk_id"]}]: {c["text"]}" for c in scores)

    return(
    f"answer the qustions using only the retrieved policy text below."
    f"if the retrived text does not contain enough information to answer,say explicitly rather than guessing .\n\n"
    f"Question: {querry}\n\n"
    f"Retrieved ploicy text : \n{context}\n\n"
    f"Answer :"
)
system_prompt="""You are an internal enterprise policy assistant.

You must answer the user's question using ONLY the policy text
provided in the current prompt.

Rules:

1. Do not use outside knowledge.
2. Do not assume information that is not explicitly stated.
3. Do not invent numbers, limits, dates, exceptions, or requirements.
4. If the provided policy text does not contain enough information
   to answer the question, explicitly state that the provided policy
   does not specify the answer.
5. If the policy contains a condition or exception, preserve it
   accurately in your answer.
6. For numerical questions, use only numbers explicitly present
   in the retrieved policy text.
7. For questions involving multiple conditions, verify every
   condition against the retrieved policy text before answering.
8. If the retrieved text is irrelevant to the question, say that
   the provided policy text does not contain the required information.
"""

def generate_rag_answer(context:str):
    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=context,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )
    return(response.text)
    

    





    



def document_search(query_topic: str) -> dict:
    """Search the policy corpus and return the best-matching chunk.

    Returns {"error": ...} when nothing scores above threshold -- this is
    what your negative_rejection tasks should trigger on purpose (see
    compensation_benefits_policy_sec6_4, which explicitly states a topic
    is NOT covered, vs. a query about something genuinely absent from the
    whole corpus, e.g. a dress code policy).
    """
    _load_index()
    query_vec = _vectorizer.transform([query_topic])
    sims = cosine_similarity(query_vec, _matrix)[0]
    best_idx = sims.argmax()

    if sims[best_idx] < 0.05:
        return {"error": "No matching policy document found"}

    chunk = _chunks[best_idx]
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "score": float(sims[best_idx]),
        "excerpt": chunk["text"],
    }


if __name__ == "__main__":
    # Smoke test -- python retrieval.py
    query="agr koi employee probation complete krny k bad overtime kam krta ha aur usy haftyyy mein 10 ghanty sa overtime krna ho to uski probation or overtime approval ke hawaly sa kia qanoon apply hon ga aur overtime rate kia ho ga"
    results = search_chunks(query)
    rag_prompt = generate_rag_prompt(scores, query)
    answer = generate_rag_answer(rag_prompt)

    print(answer)

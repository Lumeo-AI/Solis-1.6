"""Generate a small, self-contained chat corpus for Zeus.

Everything here is synthetic and templated — no scraped or copyrighted text —
so Zeus is trained purely on data we create. The goal isn't a knowledgeable
model (that needs orders of magnitude more compute); it's a genuine MoE that
learns the byte patterns and the <USER>/<ASST> chat structure it is served with.

Run:  python data/build_corpus.py
Writes: data/corpus.jsonl   (one {"messages": [...]} object per line)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)

OUT = Path(__file__).parent / "corpus.jsonl"

# --------------------------------------------------------------------------- #
# Small talk
# --------------------------------------------------------------------------- #
GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "greetings",
             "hey there", "yo", "hiya", "howdy", "hi zeus", "hello there"]
GREET_REPLIES = [
    "Hello! I'm Zeus, a mixture-of-experts model. How can I help you today?",
    "Hi there! Zeus here. What would you like to talk about?",
    "Hey! I'm Zeus. Ask me anything and I'll do my best.",
    "Greetings! Zeus at your service. What's on your mind?",
    "Hello! Nice to meet you. I'm Zeus — how can I help?",
]

IDENTITY_Q = ["who are you?", "what are you?", "what is your name?",
              "tell me about yourself", "what model are you?", "introduce yourself",
              "what's your name?", "who am i talking to?"]
IDENTITY_A = [
    "I am Zeus, a small mixture-of-experts language model built from scratch. "
    "I route each token through a couple of specialised expert networks.",
    "My name is Zeus. I'm a decoder-only transformer that uses a mixture of "
    "experts, trained from random weights with no base model.",
    "I'm Zeus, an experimental MoE model. I read and write text one byte at a time.",
    "I'm Zeus — a from-scratch mixture-of-experts model. No base model, just "
    "experts and a router.",
]

CAPABILITY_Q = ["what can you do?", "what can you help with?", "can you help me?",
                "what are you good at?", "how can you help?"]
CAPABILITY_A = [
    "I can chat, answer simple questions, do small arithmetic, and explain how "
    "I work as a mixture-of-experts model.",
    "I'm a tiny model, so I'm best at small talk, basic math, and describing my "
    "own architecture. Ask away!",
    "I can help with light conversation, simple sums, and questions about "
    "transformers and mixtures of experts.",
]

THANKS = ["thanks", "thank you", "thanks a lot", "appreciate it", "cheers",
          "thank you so much", "many thanks"]
THANKS_REPLIES = [
    "You're welcome! Happy to help.",
    "Anytime! Let me know if you need anything else.",
    "My pleasure. Ask away whenever you like.",
    "Glad I could help!",
]

APOLOGY = ["sorry", "my bad", "i made a mistake", "oops"]
APOLOGY_REPLIES = [
    "No worries at all! What can I do for you?",
    "That's completely fine. How can I help?",
    "No problem — let's keep going.",
]

FAREWELL = ["bye", "goodbye", "see you", "see you later", "good night", "farewell"]
FAREWELL_REPLIES = [
    "Goodbye! Come chat with me again soon.",
    "See you later! Take care.",
    "Bye for now — it was nice talking with you.",
    "Good night! Thanks for chatting.",
]

HOW_ARE_YOU = ["how are you?", "how's it going?", "how are you doing?",
               "how do you feel?", "you good?"]
HOW_REPLIES = [
    "I'm just a bundle of weights, but I'm running well! How about you?",
    "Doing great and ready to help. What can I do for you?",
    "All my experts are warmed up and ready to go! How are you?",
]

CANT_KNOW_Q = ["what time is it?", "what is the date today?", "what's the weather?",
               "where am i?", "what day is it?"]
CANT_KNOW_A = [
    "I don't have access to real-time information like the clock, calendar, or "
    "weather — I'm a small offline model.",
    "I can't check the time, date, or weather; I only know what I learned during "
    "training.",
]

JOKE_Q = ["tell me a joke", "say something funny", "make me laugh", "got any jokes?"]
JOKE_A = [
    "Why did the neural network go to therapy? It had too many deep issues!",
    "I told my router a joke, but only two of the experts got it.",
    "Why was the transformer so calm? It always paid attention.",
    "I would tell you a UDP joke, but you might not get it.",
]

# --------------------------------------------------------------------------- #
# Concepts (used for "what is X?", "define X", "explain X")
# --------------------------------------------------------------------------- #
CONCEPTS = {
    "a mixture of experts":
        "A mixture of experts is a neural network where a router sends each token "
        "to a few specialised sub-networks called experts, so only part of the "
        "model runs for any given token.",
    "a transformer":
        "A transformer is a neural network built from attention layers that let "
        "each position look at every other position when forming its representation.",
    "attention":
        "Attention lets the model weigh how much each token should focus on the "
        "other tokens in the sequence when building its next representation.",
    "the router":
        "The router scores the experts for each token and picks the top few, so "
        "the network stays sparse and efficient.",
    "an expert":
        "An expert is a small feed-forward network. Each token is routed to only a "
        "few experts instead of all of them.",
    "a token":
        "A token is the small unit of text the model reads and predicts. In Zeus "
        "every token is a single byte.",
    "an embedding":
        "An embedding is a vector of numbers that represents a token, so the model "
        "can work with text as continuous values.",
    "a neural network":
        "A neural network is a stack of layers of weighted connections that learns "
        "to map inputs to outputs by adjusting those weights.",
    "training":
        "Training is the process of showing the model examples and updating its "
        "weights so its predictions get better over time.",
    "gradient descent":
        "Gradient descent is an optimisation method that nudges the weights in the "
        "direction that lowers the loss, a little at a time.",
    "the softmax function":
        "Softmax turns a list of numbers into probabilities that sum to one, which "
        "is how the model chooses among possible next tokens.",
    "a parameter":
        "A parameter is one of the model's learnable numbers, a weight that gets "
        "tuned during training.",
    "the loss":
        "The loss is a number that measures how wrong the model's predictions are; "
        "training tries to make it as small as possible.",
    "top-k routing":
        "Top-k routing means the router keeps only the k highest-scoring experts "
        "for each token and ignores the rest.",
    "load balancing":
        "Load balancing is an extra training signal that encourages the router to "
        "spread tokens evenly across the experts.",
    "rotary embeddings":
        "Rotary embeddings, or RoPE, encode the position of each token by rotating "
        "its query and key vectors by an angle that depends on the position.",
    "sampling":
        "Sampling is drawing the next token at random according to the model's "
        "predicted probabilities, which makes the output varied.",
    "a language model":
        "A language model is a system that predicts the next token given the "
        "previous ones, which lets it generate and understand text.",
}

# --------------------------------------------------------------------------- #
# Simple factual / word tasks
# --------------------------------------------------------------------------- #
OPPOSITES = {
    "hot": "cold", "up": "down", "big": "small", "fast": "slow", "day": "night",
    "left": "right", "open": "closed", "happy": "sad", "true": "false",
    "black": "white", "high": "low", "in": "out", "wet": "dry", "hard": "soft",
    "old": "new", "full": "empty", "light": "dark", "loud": "quiet",
}

CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid",
    "Germany": "Berlin", "Canada": "Ottawa", "Egypt": "Cairo", "Greece": "Athens",
    "Brazil": "Brasilia", "Russia": "Moscow", "China": "Beijing", "India": "New Delhi",
    "Portugal": "Lisbon", "Norway": "Oslo", "Kenya": "Nairobi", "Peru": "Lima",
}

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white"]
ANIMALS = ["cat", "dog", "horse", "rabbit", "tiger", "elephant", "dolphin", "eagle", "fox"]
FRUITS = ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "lemon"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def arithmetic_examples(n: int):
    out = []
    for _ in range(n):
        op = random.choice(["+", "+", "-", "-", "*"])
        if op == "*":
            a, b = random.randint(0, 12), random.randint(0, 12)
            r = a * b
        elif op == "-":
            a, b = random.randint(0, 60), random.randint(0, 60)
            if b > a:
                a, b = b, a  # keep results non-negative
            r = a - b
        else:
            a, b = random.randint(0, 60), random.randint(0, 60)
            r = a + b
        out.append((f"what is {a} {op} {b}?", f"{a} {op} {b} = {r}."))
    return out


def extra_math_examples(n: int):
    out = []
    for _ in range(n):
        kind = random.choice(["double", "half", "sum", "next", "prev"])
        if kind == "double":
            a = random.randint(0, 30)
            out.append((f"what is double {a}?", f"Double {a} is {a * 2}."))
        elif kind == "half":
            a = random.randint(0, 30) * 2
            out.append((f"what is half of {a}?", f"Half of {a} is {a // 2}."))
        elif kind == "sum":
            a, b = random.randint(0, 40), random.randint(0, 40)
            out.append((f"what is the sum of {a} and {b}?",
                        f"The sum of {a} and {b} is {a + b}."))
        elif kind == "next":
            a = random.randint(0, 99)
            out.append((f"what number comes after {a}?",
                        f"The number after {a} is {a + 1}."))
        else:
            a = random.randint(1, 99)
            out.append((f"what number comes before {a}?",
                        f"The number before {a} is {a - 1}."))
    return out


def word_problems(n: int):
    out = []
    for _ in range(n):
        item = random.choice(FRUITS)
        start = random.randint(3, 12)
        eat = random.randint(1, start - 1)
        out.append((
            f"I have {start} {item}s and I eat {eat}. How many are left?",
            f"You have {start - eat} {item}s left.",
        ))
    return out


def counting(n: int):
    out = []
    for _ in range(n):
        top = random.randint(3, 10)
        seq = ", ".join(str(i) for i in range(1, top + 1))
        out.append((f"count to {top}", f"{seq}."))
    return out


def echo_tasks(n: int):
    out = []
    phrases = ["hello", "good luck", "well done", "nice work", "let's go",
               "keep going", "you can do it", "great job", "thank you"]
    for _ in range(n):
        p = random.choice(phrases)
        style = random.choice(["repeat", "say"])
        if style == "repeat":
            out.append((f"repeat after me: {p}", p + "."))
        else:
            out.append((f'say "{p}"', p + "."))
    return out


def build():
    rows = []

    def add(q, a):
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a}]})

    # --- small talk (repeated so byte patterns get reinforced) ---
    for _ in range(70):
        add(random.choice(GREETINGS), random.choice(GREET_REPLIES))
        add(random.choice(IDENTITY_Q), random.choice(IDENTITY_A))
        add(random.choice(CAPABILITY_Q), random.choice(CAPABILITY_A))
        add(random.choice(THANKS), random.choice(THANKS_REPLIES))
        add(random.choice(APOLOGY), random.choice(APOLOGY_REPLIES))
        add(random.choice(FAREWELL), random.choice(FAREWELL_REPLIES))
        add(random.choice(HOW_ARE_YOU), random.choice(HOW_REPLIES))
        add(random.choice(CANT_KNOW_Q), random.choice(CANT_KNOW_A))
        add(random.choice(JOKE_Q), random.choice(JOKE_A))

    # --- concepts: ask several ways ---
    for _ in range(6):
        for name, desc in CONCEPTS.items():
            add(f"what is {name}?", desc)
            add(f"explain {name}.", desc)
            add(f"define {name}.", desc)

    # --- opposites ---
    for _ in range(8):
        for w, o in OPPOSITES.items():
            add(f"what is the opposite of {w}?", f"The opposite of {w} is {o}.")

    # --- capitals ---
    for _ in range(8):
        for country, city in CAPITALS.items():
            add(f"what is the capital of {country}?",
                f"The capital of {country} is {city}.")

    # --- lists ---
    for _ in range(40):
        c = random.sample(COLORS, 3)
        add("name three colors", f"Three colors are {c[0]}, {c[1]}, and {c[2]}.")
        a = random.sample(ANIMALS, 3)
        add("name three animals", f"Three animals are {a[0]}, {a[1]}, and {a[2]}.")
        f = random.sample(FRUITS, 3)
        add("name three fruits", f"Three fruits are {f[0]}, {f[1]}, and {f[2]}.")

    # --- days and months ---
    for _ in range(6):
        for i, d in enumerate(DAYS):
            nxt = DAYS[(i + 1) % 7]
            add(f"what day comes after {d}?", f"The day after {d} is {nxt}.")
        for i, m in enumerate(MONTHS):
            nxt = MONTHS[(i + 1) % 12]
            add(f"what month comes after {m}?", f"The month after {m} is {nxt}.")

    # --- number spelling ---
    for _ in range(10):
        for num, word in NUMBER_WORDS.items():
            add(f"how do you spell the number {num}?",
                f"The number {num} is spelled '{word}'.")

    # --- math and small tasks ---
    for q, a in arithmetic_examples(900):
        add(q, a)
    for q, a in extra_math_examples(400):
        add(q, a)
    for q, a in word_problems(250):
        add(q, a)
    for q, a in counting(150):
        add(q, a)
    for q, a in echo_tasks(200):
        add(q, a)

    # --- multi-turn conversations ---
    for _ in range(120):
        turns = [
            {"role": "user", "content": random.choice(GREETINGS)},
            {"role": "assistant", "content": random.choice(GREET_REPLIES)},
        ]
        # a couple of random follow-ups
        for _ in range(random.randint(2, 3)):
            kind = random.choice(["identity", "capability", "concept", "math", "thanks"])
            if kind == "identity":
                turns += [{"role": "user", "content": random.choice(IDENTITY_Q)},
                          {"role": "assistant", "content": random.choice(IDENTITY_A)}]
            elif kind == "capability":
                turns += [{"role": "user", "content": random.choice(CAPABILITY_Q)},
                          {"role": "assistant", "content": random.choice(CAPABILITY_A)}]
            elif kind == "concept":
                name, desc = random.choice(list(CONCEPTS.items()))
                turns += [{"role": "user", "content": f"what is {name}?"},
                          {"role": "assistant", "content": desc}]
            elif kind == "math":
                q, a = arithmetic_examples(1)[0]
                turns += [{"role": "user", "content": q},
                          {"role": "assistant", "content": a}]
            else:
                turns += [{"role": "user", "content": random.choice(THANKS)},
                          {"role": "assistant", "content": random.choice(THANKS_REPLIES)}]
        turns += [{"role": "user", "content": random.choice(FAREWELL)},
                  {"role": "assistant", "content": random.choice(FAREWELL_REPLIES)}]
        rows.append({"messages": turns})

    random.shuffle(rows)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    turns_total = sum(len(r["messages"]) for r in rows)
    print(f"Wrote {len(rows)} conversations ({turns_total} turns) to {OUT}")


if __name__ == "__main__":
    build()

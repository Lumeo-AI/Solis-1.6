"""Generate the Solis training corpus.

Everything here is generated procedurally — no scraped or copyrighted text — so
the corpus is ours end to end, matching the from-scratch premise of the model.

The generator is weighted deliberately. A model this size cannot store facts
about the world, so teaching it trivia is wasted capacity. What it *can* learn,
and what actually makes a small model feel capable, is **in-context work**:
read the passage you were given and answer from it, transform this text, follow
this format, carry state across turns. Roughly two thirds of the corpus is
tasks of that shape, where the answer is derivable from the prompt rather than
recalled.

Run:
    python data/build_corpus.py                    # default ~200k examples
    python data/build_corpus.py --examples 500000  # bigger
    python data/build_corpus.py --seed 3 --out data/corpus_v2.jsonl

Writes JSONL, one {"messages": [...]} object per line, split into
`corpus.jsonl` (train) and `corpus.val.jsonl` (held out).
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

HERE = Path(__file__).parent
ASSISTANT_NAME = "Solis"

# --------------------------------------------------------------------------- #
# Word banks — the raw material for generated passages and tasks
# --------------------------------------------------------------------------- #
NAMES = [
    "Ada", "Bruno", "Clara", "Dmitri", "Elena", "Farid", "Greta", "Hugo",
    "Ingrid", "Jonas", "Kaito", "Lucia", "Mateo", "Nadia", "Omar", "Priya",
    "Quinn", "Rosa", "Samir", "Tomas", "Ursula", "Viktor", "Wen", "Xiomara",
    "Yusuf", "Zara", "Anika", "Bodhi", "Cleo", "Dario", "Esme", "Felix",
]
CITIES = [
    "Lisbon", "Oslo", "Kyoto", "Cairo", "Lima", "Dublin", "Prague", "Seoul",
    "Nairobi", "Bogota", "Helsinki", "Vienna", "Rabat", "Tbilisi", "Riga",
    "Valletta", "Bergen", "Porto", "Utrecht", "Ghent", "Aarhus", "Zagreb",
]
JOBS = [
    "botanist", "archivist", "welder", "cartographer", "baker", "surveyor",
    "luthier", "glassblower", "translator", "beekeeper", "typesetter",
    "ceramicist", "hydrologist", "cellist", "locksmith", "arborist",
]
OBJECTS = [
    "lantern", "compass", "ledger", "kettle", "satchel", "telescope", "anvil",
    "sundial", "typewriter", "barometer", "loom", "canoe", "harpsichord",
    "microscope", "weathervane", "abacus", "sextant", "wheelbarrow",
]
MATERIALS = ["copper", "oak", "linen", "granite", "brass", "willow", "slate",
             "cedar", "pewter", "canvas", "walnut", "iron", "silk", "clay"]
COLORS = ["red", "green", "blue", "amber", "violet", "teal", "crimson", "olive",
          "indigo", "grey", "ivory", "russet", "cobalt", "ochre"]
ADJECTIVES = [
    "quiet", "narrow", "ancient", "sturdy", "restless", "hollow", "brittle",
    "luminous", "cramped", "sprawling", "meticulous", "weathered", "fragrant",
    "cluttered", "spartan", "humid", "windswept", "orderly", "battered",
]
ANIMALS = ["heron", "otter", "badger", "falcon", "tortoise", "lynx", "magpie",
           "ibex", "pelican", "marten", "kestrel", "hare", "newt", "gannet"]
FRUITS = ["apple", "pear", "fig", "plum", "quince", "apricot", "medlar",
          "damson", "greengage", "nectarine", "persimmon", "mulberry"]
VERBS_PAST = ["repaired", "catalogued", "sold", "carried", "polished", "traded",
              "sketched", "measured", "hauled", "wrapped", "inherited", "rebuilt"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# --------------------------------------------------------------------------- #
# Passage synthesis — the substrate for in-context tasks
# --------------------------------------------------------------------------- #
def make_passage(rng: random.Random, n_sentences: int = 5) -> tuple[str, dict]:
    """Build a short passage plus a dict of ground-truth facts about it.

    Because we generate the facts first and the prose second, we can ask
    questions whose answers are guaranteed correct and guaranteed present in the
    text — which is exactly the supervision an extractive-QA skill needs.
    """
    person = rng.choice(NAMES)
    city = rng.choice(CITIES)
    job = rng.choice(JOBS)
    obj = rng.choice(OBJECTS)
    material = rng.choice(MATERIALS)
    color = rng.choice(COLORS)
    animal = rng.choice(ANIMALS)
    count = rng.randint(2, 40)
    year = rng.randint(1890, 2020)
    age = rng.randint(19, 78)
    price = rng.randint(3, 500)
    day = rng.choice(WEEKDAYS)

    facts = {
        "person": person, "city": city, "job": job, "object": obj,
        "material": material, "color": color, "animal": animal,
        "count": count, "year": year, "age": age, "price": price, "day": day,
    }

    pool = [
        f"{person} works as a {job} in {city}.",
        f"{person} is {age} years old.",
        f"The workshop was founded in {year}.",
        f"{person} owns {count} {obj}s.",
        f"Each {obj} is made of {material}.",
        f"The {obj} on the shelf is {color}.",
        f"A {animal} nests above the door.",
        f"{person} sells each {obj} for {price} coins.",
        f"The shop is closed on {day}.",
        f"{person} {rng.choice(VERBS_PAST)} the {obj} last spring.",
    ]
    rng.shuffle(pool)
    sentences = pool[:max(3, min(n_sentences, len(pool)))]
    return " ".join(sentences), {**facts, "sentences": sentences}


def passage_questions(rng: random.Random, passage: str, facts: dict
                      ) -> list[tuple[str, str]]:
    """Questions answerable *only* by reading the passage."""
    qs: list[tuple[str, str]] = []
    s = facts["sentences"]
    text = " ".join(s)

    if f"works as a {facts['job']}" in text:
        qs.append((f"What does {facts['person']} do for a living?",
                   f"{facts['person']} works as a {facts['job']}."))
        qs.append((f"Where does {facts['person']} work?",
                   f"{facts['person']} works in {facts['city']}."))
    if f"is {facts['age']} years old" in text:
        qs.append((f"How old is {facts['person']}?",
                   f"{facts['person']} is {facts['age']} years old."))
    if f"founded in {facts['year']}" in text:
        qs.append(("In what year was the workshop founded?",
                   f"The workshop was founded in {facts['year']}."))
    if f"owns {facts['count']}" in text:
        qs.append((f"How many {facts['object']}s does {facts['person']} own?",
                   f"{facts['person']} owns {facts['count']} "
                   f"{facts['object']}s."))
    if f"made of {facts['material']}" in text:
        qs.append((f"What are the {facts['object']}s made of?",
                   f"They are made of {facts['material']}."))
    if f"is {facts['color']}" in text:
        qs.append((f"What colour is the {facts['object']} on the shelf?",
                   f"It is {facts['color']}."))
    if f"A {facts['animal']} nests" in text:
        qs.append(("What animal nests above the door?",
                   f"A {facts['animal']} nests above the door."))
    if f"for {facts['price']} coins" in text:
        qs.append((f"How much does one {facts['object']} cost?",
                   f"Each {facts['object']} costs {facts['price']} coins."))
        if f"owns {facts['count']}" in text:
            total = facts["count"] * facts["price"]
            qs.append((
                f"If {facts['person']} sold every {facts['object']}, how much "
                "would that be?",
                f"{facts['count']} x {facts['price']} = {total} coins."))
    if f"closed on {facts['day']}" in text:
        qs.append(("Which day is the shop closed?",
                   f"The shop is closed on {facts['day']}."))

    # A question the passage genuinely does not answer — teaches the model to
    # decline instead of inventing, which is most of what "trustworthy" means
    # at this scale.
    qs.append((rng.choice([
        "What is the owner's phone number?",
        "How many windows does the shop have?",
        "What did they eat for breakfast?",
        "Who lives next door?",
    ]), "The passage doesn't say."))
    return qs


# --------------------------------------------------------------------------- #
# Task generators. Each returns (user_text, assistant_text).
# --------------------------------------------------------------------------- #
def t_passage_qa(rng):
    passage, facts = make_passage(rng, rng.randint(4, 7))
    qs = passage_questions(rng, passage, facts)
    q, a = rng.choice(qs)
    lead = rng.choice([
        f"Read this and answer the question.\n\n{passage}\n\nQuestion: {q}",
        f"{passage}\n\n{q}",
        f"Based only on the text below, {q[0].lower()}{q[1:]}\n\n{passage}",
    ])
    return lead, a


def t_passage_multi_qa(rng):
    """Several questions at once — teaches multi-part answers and formatting."""
    passage, facts = make_passage(rng, rng.randint(5, 8))
    qs = passage_questions(rng, passage, facts)
    picked = rng.sample(qs, min(3, len(qs)))
    body = "\n".join(f"{i + 1}. {q}" for i, (q, _) in enumerate(picked))
    ans = "\n".join(f"{i + 1}. {a}" for i, (_, a) in enumerate(picked))
    return f"{passage}\n\nAnswer these:\n{body}", ans


def t_summarize(rng):
    passage, facts = make_passage(rng, rng.randint(5, 8))
    summary = (f"{facts['person']} is a {facts['job']} in {facts['city']} who "
               f"owns {facts['count']} {facts['object']}s.")
    return (f"Summarise this in one sentence.\n\n{passage}", summary)


def t_extract_json(rng):
    passage, facts = make_passage(rng, 6)
    keys = ["person", "city", "job"]
    obj = {k: facts[k] for k in keys}
    return (f"Extract the person, city, and job from this text as JSON.\n\n"
            f"{passage}", json.dumps(obj))


def t_text_transform(rng):
    words = rng.sample(ADJECTIVES + ANIMALS + OBJECTS + FRUITS, rng.randint(4, 9))
    phrase = " ".join(words)
    kind = rng.choice([
        "upper", "lower", "title", "reverse_words", "reverse_chars", "sort",
        "count_words", "count_chars", "first", "last", "join_commas",
        "longest", "shortest", "initials", "dedupe",
    ])
    if kind == "upper":
        return f"Convert to uppercase: {phrase}", phrase.upper()
    if kind == "lower":
        return f"Convert to lowercase: {phrase.upper()}", phrase
    if kind == "title":
        return f"Put this in title case: {phrase}", phrase.title()
    if kind == "reverse_words":
        return (f"Reverse the order of these words: {phrase}",
                " ".join(reversed(words)))
    if kind == "reverse_chars":
        w = rng.choice(words)
        return f"Spell '{w}' backwards.", w[::-1]
    if kind == "sort":
        return (f"Sort these words alphabetically: {phrase}",
                " ".join(sorted(words)))
    if kind == "count_words":
        return f"How many words are in this? {phrase}", f"There are {len(words)} words."
    if kind == "count_chars":
        w = rng.choice(words)
        return f"How many letters are in '{w}'?", f"'{w}' has {len(w)} letters."
    if kind == "first":
        return f"What is the first word? {phrase}", f"The first word is '{words[0]}'."
    if kind == "last":
        return f"What is the last word? {phrase}", f"The last word is '{words[-1]}'."
    if kind == "join_commas":
        return (f"Rewrite this list with commas: {phrase}", ", ".join(words))
    if kind == "longest":
        best = max(words, key=len)
        return f"Which word is longest? {phrase}", f"The longest word is '{best}'."
    if kind == "shortest":
        best = min(words, key=len)
        return f"Which word is shortest? {phrase}", f"The shortest word is '{best}'."
    if kind == "initials":
        return (f"Give the initials of these words: {phrase}",
                "".join(w[0].upper() for w in words))
    dupes = words + rng.sample(words, 2)
    rng.shuffle(dupes)
    seen, out = set(), []
    for w in dupes:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return f"Remove the duplicates: {' '.join(dupes)}", " ".join(out)


def t_list_format(rng):
    items = rng.sample(OBJECTS + ANIMALS + FRUITS, rng.randint(3, 6))
    style = rng.choice(["bullets", "numbered", "json", "csv", "lines"])
    if style == "bullets":
        return ("Turn this into a bulleted list: " + ", ".join(items),
                "\n".join(f"- {i}" for i in items))
    if style == "numbered":
        return ("Turn this into a numbered list: " + ", ".join(items),
                "\n".join(f"{n}. {i}" for n, i in enumerate(items, 1)))
    if style == "json":
        return ("Give me this as a JSON array: " + ", ".join(items),
                json.dumps(items))
    if style == "csv":
        return ("Format as CSV on one line: " + " ".join(items), ",".join(items))
    return ("Put each on its own line: " + ", ".join(items), "\n".join(items))


def t_arithmetic(rng):
    """Arithmetic with the working shown — the reasoning trace is the point."""
    kind = rng.choice(["add", "sub", "mul", "div", "chain", "percent", "avg"])
    if kind == "add":
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        return f"What is {a} + {b}?", f"{a} + {b} = {a + b}."
    if kind == "sub":
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        if b > a:
            a, b = b, a
        return f"What is {a} - {b}?", f"{a} - {b} = {a - b}."
    if kind == "mul":
        a, b = rng.randint(0, 40), rng.randint(0, 40)
        return f"What is {a} * {b}?", f"{a} * {b} = {a * b}."
    if kind == "div":
        b = rng.randint(2, 20)
        r = rng.randint(1, 40)
        a = b * r
        return f"What is {a} / {b}?", f"{a} / {b} = {r}."
    if kind == "chain":
        a, b, c = (rng.randint(1, 30) for _ in range(3))
        step = a + b
        return (f"What is {a} + {b} * {c}?",
                f"Multiplication first: {b} * {c} = {b * c}. "
                f"Then {a} + {b * c} = {a + b * c}.")
    if kind == "percent":
        pct = rng.choice([10, 20, 25, 50, 75])
        base = rng.randint(1, 40) * 4
        return (f"What is {pct}% of {base}?",
                f"{pct}% of {base} = {base * pct // 100}.")
    nums = [rng.randint(1, 50) for _ in range(rng.randint(3, 5))]
    total = sum(nums)
    if total % len(nums) != 0:
        nums[-1] += len(nums) - (total % len(nums))
        total = sum(nums)
    return (f"What is the average of {', '.join(map(str, nums))}?",
            f"The sum is {total} and there are {len(nums)} numbers, "
            f"so the average is {total // len(nums)}.")


def t_word_problem(rng):
    person = rng.choice(NAMES)
    item = rng.choice(FRUITS)
    kind = rng.choice(["buy_eat", "rate", "split", "compare", "total"])
    if kind == "buy_eat":
        start, eat = rng.randint(5, 40), rng.randint(1, 4)
        return (f"{person} has {start} {item}s and gives away {eat}. "
                "How many are left?",
                f"{start} - {eat} = {start - eat}. "
                f"{person} has {start - eat} {item}s left.")
    if kind == "rate":
        per, days = rng.randint(2, 9), rng.randint(2, 9)
        return (f"{person} picks {per} {item}s a day for {days} days. "
                "How many is that?",
                f"{per} x {days} = {per * days} {item}s.")
    if kind == "split":
        people = rng.randint(2, 8)
        each = rng.randint(2, 12)
        total = people * each
        return (f"{total} {item}s are shared equally among {people} friends. "
                "How many each?",
                f"{total} / {people} = {each}. Each friend gets {each}.")
    if kind == "compare":
        other = rng.choice([n for n in NAMES if n != person])
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        while a == b:
            b = rng.randint(2, 60)
        winner, diff = (person, a - b) if a > b else (other, b - a)
        return (f"{person} has {a} {item}s and {other} has {b}. Who has more, "
                "and by how many?",
                f"{winner} has more, by {diff}.")
    price, qty = rng.randint(2, 30), rng.randint(2, 12)
    return (f"Each {item} costs {price} coins. What do {qty} cost?",
            f"{price} x {qty} = {price * qty} coins.")


def t_sequence(rng):
    kind = rng.choice(["arith", "double", "square", "fib", "count"])
    if kind == "arith":
        start, step = rng.randint(1, 20), rng.randint(2, 12)
        seq = [start + i * step for i in range(5)]
        return (f"What comes next? {', '.join(map(str, seq))}",
                f"The step is {step}, so the next number is {seq[-1] + step}.")
    if kind == "double":
        start = rng.randint(1, 12)
        seq = [start * 2 ** i for i in range(5)]
        return (f"What comes next? {', '.join(map(str, seq))}",
                f"Each number doubles, so the next is {seq[-1] * 2}.")
    if kind == "square":
        n = rng.randint(1, 6)
        seq = [(n + i) ** 2 for i in range(4)]
        return (f"What comes next? {', '.join(map(str, seq))}",
                f"These are squares, so the next is {(n + 4) ** 2}.")
    if kind == "fib":
        a, b = rng.randint(1, 5), rng.randint(1, 5)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
        return (f"What comes next? {', '.join(map(str, seq))}",
                f"Each number is the sum of the previous two, so the next is "
                f"{seq[-1] + seq[-2]}.")
    lo = rng.randint(1, 20)
    hi = lo + rng.randint(3, 9)
    return (f"Count from {lo} to {hi}.",
            ", ".join(str(i) for i in range(lo, hi + 1)) + ".")


def t_code(rng):
    kind = rng.choice(["write", "explain", "output", "fix"])
    fname = rng.choice(["add", "double", "square", "greet", "is_even",
                        "count_items", "last_item", "shout"])
    bodies = {
        "add": ("def add(a, b):\n    return a + b", "adds two numbers"),
        "double": ("def double(x):\n    return x * 2", "doubles a number"),
        "square": ("def square(x):\n    return x * x", "squares a number"),
        "greet": ('def greet(name):\n    return f"Hello, {name}!"',
                  "builds a greeting string"),
        "is_even": ("def is_even(n):\n    return n % 2 == 0",
                    "checks whether a number is even"),
        "count_items": ("def count_items(items):\n    return len(items)",
                        "returns how many items a list has"),
        "last_item": ("def last_item(items):\n    return items[-1]",
                      "returns the last element of a list"),
        "shout": ("def shout(text):\n    return text.upper()",
                  "uppercases a string"),
    }
    code, desc = bodies[fname]
    if kind == "write":
        return (f"Write a Python function called {fname} that {desc}.",
                f"```python\n{code}\n```")
    if kind == "explain":
        return (f"What does this do?\n\n```python\n{code}\n```",
                f"It {desc}.")
    if kind == "fix":
        broken = code.replace("return", "retrun", 1)
        return (f"There's a bug here, fix it.\n\n```python\n{broken}\n```",
                f"`retrun` should be `return`.\n\n```python\n{code}\n```")
    # Predict the output.
    if fname == "add":
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        return (f"```python\n{code}\nprint(add({a}, {b}))\n```\nWhat prints?",
                str(a + b))
    if fname == "double":
        a = rng.randint(1, 50)
        return (f"```python\n{code}\nprint(double({a}))\n```\nWhat prints?",
                str(a * 2))
    if fname == "square":
        a = rng.randint(1, 20)
        return (f"```python\n{code}\nprint(square({a}))\n```\nWhat prints?",
                str(a * a))
    if fname == "is_even":
        a = rng.randint(1, 50)
        return (f"```python\n{code}\nprint(is_even({a}))\n```\nWhat prints?",
                str(a % 2 == 0))
    name = rng.choice(NAMES)
    return (f"```python\n{code}\nprint(greet(\"{name}\"))\n```\nWhat prints?",
            f"Hello, {name}!")


def t_logic(rng):
    kind = rng.choice(["order", "compare", "syllogism", "negation"])
    a, b, c = rng.sample(NAMES, 3)
    if kind == "order":
        return (f"{a} is taller than {b}. {b} is taller than {c}. "
                "Who is shortest?", f"{c} is the shortest.")
    if kind == "compare":
        x, y = rng.sample(OBJECTS, 2)
        return (f"The {x} is heavier than the {y}. Which is lighter?",
                f"The {y} is lighter.")
    if kind == "syllogism":
        animal = rng.choice(ANIMALS)
        return (f"All {animal}s are shy. Pip is a {animal}. Is Pip shy?",
                "Yes, Pip is shy.")
    day = rng.choice(WEEKDAYS)
    return (f"The shop is open every day except {day}. Is it open on {day}?",
            f"No, it is closed on {day}.")


def t_calendar(rng):
    kind = rng.choice(["next_day", "prev_day", "next_month", "days_in", "nth"])
    if kind == "next_day":
        i = rng.randrange(7)
        return (f"What day comes after {WEEKDAYS[i]}?",
                f"{WEEKDAYS[(i + 1) % 7]}.")
    if kind == "prev_day":
        i = rng.randrange(7)
        return (f"What day comes before {WEEKDAYS[i]}?",
                f"{WEEKDAYS[(i - 1) % 7]}.")
    if kind == "next_month":
        i = rng.randrange(12)
        return (f"Which month follows {MONTHS[i]}?", f"{MONTHS[(i + 1) % 12]}.")
    if kind == "days_in":
        i = rng.randrange(12)
        return (f"How many days are in {MONTHS[i]}?",
                f"{MONTHS[i]} has {MONTH_DAYS[i]} days"
                + (" in a common year." if i == 1 else "."))
    n = rng.randint(1, 7)
    return (f"What is day {n} of the week, starting on Monday?",
            f"{WEEKDAYS[n - 1]}.")


def t_units(rng):
    kind = rng.choice(["len", "mass", "time", "vol"])
    if kind == "len":
        n = rng.randint(1, 90)
        return f"How many centimetres are in {n} metres?", f"{n * 100} cm."
    if kind == "mass":
        n = rng.randint(1, 90)
        return f"How many grams are in {n} kilograms?", f"{n * 1000} g."
    if kind == "time":
        n = rng.randint(1, 24)
        return f"How many minutes are in {n} hours?", f"{n * 60} minutes."
    n = rng.randint(1, 40)
    return f"How many millilitres are in {n} litres?", f"{n * 1000} ml."


def t_identity(rng):
    q = rng.choice([
        "who are you?", "what are you?", "what's your name?",
        "introduce yourself", "what model are you?", "tell me about yourself",
        "who am i talking to?", "what kind of model is this?",
    ])
    a = rng.choice([
        f"I'm {ASSISTANT_NAME}, a sparse mixture-of-experts language model built "
        "from scratch. A router sends each token through a few specialist "
        "networks instead of the whole model, so I stay fast.",
        f"My name is {ASSISTANT_NAME}. I'm a decoder-only transformer with a "
        "mixture-of-experts feed-forward layer, trained from random weights — "
        "no base model underneath.",
        f"I'm {ASSISTANT_NAME}. I'm a small model, so I'm best at working with "
        "text you give me rather than recalling facts on my own.",
        f"I'm {ASSISTANT_NAME}, an experimental MoE model. Every token I read "
        "goes through a shared expert plus a handful of routed ones.",
    ])
    return q, a


def t_capability(rng):
    q = rng.choice([
        "what can you do?", "what are you good at?", "how can you help?",
        "what should I ask you?", "what are your limits?",
    ])
    a = rng.choice([
        "I'm strongest on text you hand me: answering questions about a "
        "passage, reformatting, summarising, extracting fields, and small "
        "arithmetic. I'm weak on facts I'd have to recall from memory.",
        "Give me something to work with — a paragraph, a list, some numbers — "
        "and I'll read it, transform it, or answer questions about it. Ask me "
        "for world facts and I'll often be wrong.",
        "Good at: reading comprehension, formatting, step-by-step arithmetic, "
        "simple code. Bad at: current events, obscure trivia, long chains of "
        "reasoning. I'm a small model and I'd rather tell you that up front.",
    ])
    return q, a


def t_calibration(rng):
    """Teach the model to decline rather than confabulate."""
    q = rng.choice([
        "what's the weather today?", "what time is it?", "what's today's date?",
        "what's in the news?", "what's my name?", "look up this website for me",
        "what did I say yesterday?", "what's the current stock price?",
    ])
    a = rng.choice([
        "I can't check that — I have no clock, no internet, and no memory "
        "between conversations.",
        "I don't have access to real-time information, so I can't answer that "
        "accurately.",
        "That's outside what I can see. I only know this conversation and what "
        "I learned in training.",
    ])
    return q, a


def t_smalltalk(rng):
    pairs = [
        (["hi", "hello", "hey", "yo", "good morning", "hey there", "howdy"],
         [f"Hello! I'm {ASSISTANT_NAME}. What are we working on?",
          f"Hi there — {ASSISTANT_NAME} here. What can I do for you?",
          "Hey! Give me some text or a question and I'll get going.",
          f"Hello! {ASSISTANT_NAME} at your service."]),
        (["thanks", "thank you", "cheers", "appreciate it", "thanks a lot"],
         ["You're welcome.", "Anytime — anything else?",
          "Happy to help.", "My pleasure."]),
        (["bye", "goodbye", "see you", "good night", "later"],
         ["Goodbye!", "See you later.", "Take care.",
          "Bye for now — come back any time."]),
        (["how are you?", "how's it going?", "you good?"],
         ["Running well, thanks. What do you need?",
          "All experts warmed up. What's the task?",
          "I'm a set of weights, so: unchanged. What can I do?"]),
        (["sorry", "my bad", "oops", "I made a mistake"],
         ["No problem at all.", "That's fine — let's keep going.",
          "No worries. What next?"]),
    ]
    qs, as_ = rng.choice(pairs)
    return rng.choice(qs), rng.choice(as_)


def t_definition(rng):
    concepts = {
        "a mixture of experts": "A neural network where a router sends each "
            "token to a few specialised sub-networks, so only part of the model "
            "runs for any given token.",
        "a transformer": "A neural network built from attention layers, where "
            "each position can look at every other position when forming its "
            "representation.",
        "attention": "A mechanism that weighs how much each token should focus "
            "on the other tokens in the sequence.",
        "a router": "The small network that scores the experts for each token "
            "and picks the top few.",
        "a token": "The unit of text a model reads and predicts — usually a "
            "word piece rather than a whole word.",
        "an embedding": "A vector of numbers representing a token, so the model "
            "can work with text as continuous values.",
        "training": "Showing the model examples and adjusting its weights so its "
            "predictions improve.",
        "gradient descent": "An optimisation method that nudges weights in the "
            "direction that lowers the loss.",
        "softmax": "A function that turns a list of numbers into probabilities "
            "that sum to one.",
        "a parameter": "One of the model's learnable numbers.",
        "loss": "A number measuring how wrong the model's predictions are; "
            "training tries to make it small.",
        "top-k routing": "Keeping only the k highest-scoring experts for each "
            "token and ignoring the rest.",
        "load balancing": "An extra training signal that spreads tokens evenly "
            "across the experts so none go unused.",
        "rotary embeddings": "A way of encoding position by rotating each "
            "token's query and key vectors by a position-dependent angle.",
        "a KV cache": "Stored keys and values from earlier tokens, so generating "
            "the next token doesn't require re-reading the whole sequence.",
        "grouped-query attention": "Sharing one set of keys and values across "
            "several query heads, which shrinks the KV cache.",
        "quantisation": "Storing weights at lower precision so a model needs "
            "less memory.",
        "perplexity": "The exponential of the average loss — roughly, how many "
            "options the model is choosing between at each token.",
        "a context window": "The maximum number of tokens a model can consider "
            "at once.",
        "sampling": "Drawing the next token at random from the model's "
            "predicted probabilities, which makes output varied.",
    }
    name, desc = rng.choice(list(concepts.items()))
    q = rng.choice([f"What is {name}?", f"Explain {name}.", f"Define {name}.",
                    f"What does {name} mean?"])
    return q, desc


def t_grammar(rng):
    kind = rng.choice(["plural", "past", "opposite", "article"])
    if kind == "plural":
        w = rng.choice(["box", "city", "leaf", "child", "mouse", "wolf",
                        "bus", "party", "knife", "goose"])
        plurals = {"box": "boxes", "city": "cities", "leaf": "leaves",
                   "child": "children", "mouse": "mice", "wolf": "wolves",
                   "bus": "buses", "party": "parties", "knife": "knives",
                   "goose": "geese"}
        return f"What is the plural of '{w}'?", f"The plural of '{w}' is '{plurals[w]}'."
    if kind == "past":
        w = rng.choice(["go", "run", "eat", "write", "bring", "teach",
                        "swim", "buy", "think", "catch"])
        past = {"go": "went", "run": "ran", "eat": "ate", "write": "wrote",
                "bring": "brought", "teach": "taught", "swim": "swam",
                "buy": "bought", "think": "thought", "catch": "caught"}
        return f"What is the past tense of '{w}'?", f"'{past[w]}'."
    if kind == "opposite":
        pairs = {"hot": "cold", "up": "down", "big": "small", "fast": "slow",
                 "open": "closed", "happy": "sad", "wet": "dry", "hard": "soft",
                 "light": "dark", "loud": "quiet", "early": "late",
                 "empty": "full", "rough": "smooth", "sharp": "blunt"}
        w = rng.choice(list(pairs))
        return f"What is the opposite of '{w}'?", f"The opposite of '{w}' is '{pairs[w]}'."
    w = rng.choice(ANIMALS + OBJECTS)
    art = "an" if w[0] in "aeiou" else "a"
    return f"Should it be 'a' or 'an' before '{w}'?", f"'{art} {w}'."


def t_table(rng):
    n = rng.randint(2, 4)
    rows = [(rng.choice(NAMES), rng.choice(CITIES), rng.randint(20, 70))
            for _ in range(n)]
    lines = "\n".join(f"{a}, {b}, {c}" for a, b, c in rows)
    kind = rng.choice(["md", "oldest", "count", "cities"])
    if kind == "md":
        head = "| Name | City | Age |\n| --- | --- | --- |"
        body = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
        return (f"Turn this into a markdown table with columns Name, City, "
                f"Age.\n\n{lines}", f"{head}\n{body}")
    if kind == "oldest":
        oldest = max(rows, key=lambda r: r[2])
        return (f"Who is oldest?\n\n{lines}",
                f"{oldest[0]}, at {oldest[2]}.")
    if kind == "count":
        return f"How many rows are there?\n\n{lines}", f"{n} rows."
    return (f"List just the cities.\n\n{lines}",
            ", ".join(r[1] for r in rows))


def t_rewrite(rng):
    person = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    city = rng.choice(CITIES)
    sentence = f"{person} bought a {obj} in {city}."
    kind = rng.choice(["question", "negate", "past_to_future", "polite"])
    if kind == "question":
        return (f"Turn this into a question: {sentence}",
                f"Did {person} buy a {obj} in {city}?")
    if kind == "negate":
        return (f"Make this negative: {sentence}",
                f"{person} did not buy a {obj} in {city}.")
    if kind == "past_to_future":
        return (f"Rewrite this in the future tense: {sentence}",
                f"{person} will buy a {obj} in {city}.")
    return (f"Make this more polite: give me the {obj}",
            f"Could you pass me the {obj}, please?")


def t_spelling(rng):
    w = rng.choice(OBJECTS + ANIMALS + FRUITS)
    kind = rng.choice(["spell", "letter_n", "starts", "vowels"])
    if kind == "spell":
        return f"Spell '{w}' one letter at a time.", "-".join(w.upper())
    if kind == "letter_n":
        n = rng.randint(1, len(w))
        return (f"What is letter {n} of '{w}'?",
                f"Letter {n} of '{w}' is '{w[n - 1]}'.")
    if kind == "starts":
        return f"What letter does '{w}' start with?", f"'{w}' starts with '{w[0]}'."
    v = sum(1 for ch in w if ch in "aeiou")
    return f"How many vowels are in '{w}'?", f"'{w}' has {v} vowels."


# --------------------------------------------------------------------------- #
# Multi-turn conversation assembly
# --------------------------------------------------------------------------- #
SINGLE_TURN = [
    (t_passage_qa, 14),
    (t_passage_multi_qa, 5),
    (t_summarize, 4),
    (t_extract_json, 4),
    (t_text_transform, 12),
    (t_list_format, 6),
    (t_arithmetic, 10),
    (t_word_problem, 7),
    (t_sequence, 4),
    (t_code, 6),
    (t_logic, 4),
    (t_calendar, 3),
    (t_units, 2),
    (t_table, 4),
    (t_rewrite, 3),
    (t_spelling, 3),
    (t_grammar, 3),
    (t_definition, 4),
    (t_identity, 3),
    (t_capability, 2),
    (t_calibration, 3),
    (t_smalltalk, 4),
]
_TASKS = [fn for fn, w in SINGLE_TURN for _ in range(w)]


def make_single(rng) -> dict:
    q, a = rng.choice(_TASKS)(rng)
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def make_multi_turn(rng) -> dict:
    """A conversation whose later turns depend on earlier ones.

    This is the only place the model can learn to resolve "it", "that one", or
    "the same thing again" — none of which appear in single-turn data.
    """
    msgs = []
    if rng.random() < 0.5:
        q, a = t_smalltalk(rng)
        msgs += [{"role": "user", "content": q},
                 {"role": "assistant", "content": a}]

    style = rng.choice(["passage_thread", "mixed", "arith_thread"])

    if style == "passage_thread":
        # One passage, several follow-up questions about it — the model must
        # keep referring back to text from many turns ago.
        passage, facts = make_passage(rng, rng.randint(5, 8))
        qs = passage_questions(rng, passage, facts)
        rng.shuffle(qs)
        first_q, first_a = qs[0]
        msgs += [
            {"role": "user", "content": f"{passage}\n\n{first_q}"},
            {"role": "assistant", "content": first_a},
        ]
        for q, a in qs[1:rng.randint(2, 4)]:
            msgs += [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}]

    elif style == "arith_thread":
        total = rng.randint(10, 60)
        msgs += [
            {"role": "user", "content": f"What is {total} + {total}?"},
            {"role": "assistant", "content": f"{total} + {total} = {total * 2}."},
            {"role": "user", "content": "Now double that."},
            {"role": "assistant", "content": f"{total * 2} x 2 = {total * 4}."},
            {"role": "user", "content": "And subtract ten."},
            {"role": "assistant", "content": f"{total * 4} - 10 = {total * 4 - 10}."},
        ]

    else:
        for _ in range(rng.randint(2, 4)):
            q, a = rng.choice(_TASKS)(rng)
            msgs += [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}]

    if rng.random() < 0.35:
        q, a = t_smalltalk(rng)
        msgs += [{"role": "user", "content": q},
                 {"role": "assistant", "content": a}]
    return {"messages": msgs}


SYSTEM_PROMPTS = [
    f"You are {ASSISTANT_NAME}, a helpful assistant.",
    f"You are {ASSISTANT_NAME}. Answer concisely.",
    f"You are {ASSISTANT_NAME}, a small mixture-of-experts model. Be direct and "
    "say when you don't know.",
    "Answer using only the information given to you.",
    "Be brief. Prefer lists when listing things.",
]


def build(n_examples: int, seed: int, val_fraction: float,
          out: Path, multi_turn_fraction: float = 0.25,
          system_fraction: float = 0.3):
    rng = random.Random(seed)
    rows = []
    for i in range(n_examples):
        row = (make_multi_turn(rng) if rng.random() < multi_turn_fraction
               else make_single(rng))
        if rng.random() < system_fraction:
            row["messages"].insert(
                0, {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)})
        rows.append(row)
        if (i + 1) % 50_000 == 0:
            print(f"  generated {i + 1:,}/{n_examples:,}")

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val, train = rows[:n_val], rows[n_val:]

    out.parent.mkdir(parents=True, exist_ok=True)
    val_path = out.with_suffix(".val.jsonl")

    def dump(path: Path, data):
        with path.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path.stat().st_size

    train_bytes = dump(out, train)
    val_bytes = dump(val_path, val)

    chars = sum(len(m["content"]) for r in rows for m in r["messages"])
    turns = sum(len(r["messages"]) for r in rows)
    print(f"\ntrain: {len(train):,} conversations  ({train_bytes / 1e6:.1f} MB) "
          f"-> {out}")
    print(f"val:   {len(val):,} conversations  ({val_bytes / 1e6:.1f} MB) "
          f"-> {val_path}")
    print(f"total: {turns:,} turns, {chars / 1e6:.1f}M characters")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=int, default=200_000,
                    help="number of conversations to generate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--multi-turn-fraction", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=HERE / "corpus.jsonl")
    args = ap.parse_args()

    print(f"generating {args.examples:,} conversations (seed {args.seed})")
    build(args.examples, args.seed, args.val_fraction, args.out,
          args.multi_turn_fraction)


if __name__ == "__main__":
    main()

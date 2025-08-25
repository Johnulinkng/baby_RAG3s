import time
from babycare_rag import BabyCareRAG

rag = BabyCareRAG()

questions = [
    "What are the ABCs of Safe Sleep?",
    "What is the ideal room temperature for a baby's nursery?",
    "What are the best programming languages for children to learn?"
]

for q in questions:
    t0 = time.perf_counter()
    resp = rag.query(q)
    t1 = time.perf_counter()
    print({
        'question': q,
        'elapsed_sec': round(t1 - t0, 3),
        'sources': resp.sources[:2],
        'answer_preview': resp.answer[:100].replace('\n',' ')
    })


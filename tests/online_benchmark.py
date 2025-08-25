import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure we load env from .env if present
load_dotenv()

from babycare_rag.api import BabyCareRAGAPI


def run_benchmark(questions: list[str], max_steps: int = 3):
    api = BabyCareRAGAPI()
    durations = []

    print(f"Running {len(questions)} queries...")
    for q in questions:
        t1 = time.time()
        res = api.query(q, max_steps=max_steps)
        t2 = time.time()
        dt = round(t2 - t1, 3)
        durations.append(dt)
        status = "OK" if res.get("success") else "ERR"
        print(f"- {q[:18]}... -> {status}  time={dt}s")
    avg = round(sum(durations) / len(durations), 3) if durations else 0.0
    print(f"Average: {avg}s")


if __name__ == "__main__":
    qs = [
        "婴儿房间的理想温度是多少？",
        "如何安抚哭闹的婴儿？",
        "什么时候开始添加辅食？",
    ]
    run_benchmark(qs)


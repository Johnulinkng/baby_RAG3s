import re
from typing import List, Dict

def extract_temperature(text: str) -> List[Dict]:
    patterns = [
        r"(\d+)\s*[-~至到]\s*(\d+)\s*[°]?[Cc]",
        r"(\d+)\s*[-~至到]\s*(\d+)\s*[°]?[Ff]",
    ]
    results: List[Dict] = []
    for p in patterns:
        for match in re.finditer(p, text):
            min_val, max_val = float(match.group(1)), float(match.group(2))
            unit = 'C' if 'C' in p or 'c' in p else 'F'
            results.append({'min': min_val, 'max': max_val, 'unit': unit, 'source_text': match.group(0)})
    return results


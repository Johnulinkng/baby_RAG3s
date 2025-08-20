import re
from typing import List, Dict

def extract_temperature(text: str) -> List[Dict]:
    # Patterns paired with fixed unit to avoid relying on text content
    patterns_with_unit = [
        # Basic range patterns
        (r"(\d+)\s*[-~至到]\s*(\d+)\s*[°]?[Cc]", 'C'),
        (r"(\d+)\s*[-~至到]\s*(\d+)\s*[°]?[Ff]", 'F'),
        (r"(\d+)\s*[-~至到]\s*(\d+)\s*(?:度)?\s*(?:Celsius|celsius)", 'C'),
        (r"(\d+)\s*[-~至到]\s*(\d+)\s*(?:度)?\s*(?:Fahrenheit|fahrenheit)", 'F'),
        # English "to" patterns
        (r"(\d+)\s+to\s+(\d+)\s*(?:degrees?)?\s*[Ff]", 'F'),
        (r"(\d+)\s+to\s+(\d+)\s*(?:degrees?)?\s*[Cc]", 'C'),
        (r"(\d+)\s+to\s+(\d+)\s*(?:degrees?)?\s*(?:Fahrenheit|fahrenheit)", 'F'),
        (r"(\d+)\s+to\s+(\d+)\s*(?:degrees?)?\s*(?:Celsius|celsius)", 'C'),
        # "around X to Y" patterns
        (r"around\s+(\d+)\s+to\s+(\d+)\s*(?:degrees?)?\s*[Ff]", 'F'),
        (r"around\s+(\d+)\s+to\s+(\d+)\s*(?:degrees?)?\s*[Cc]", 'C'),
    ]
    results: List[Dict] = []
    for p, unit in patterns_with_unit:
        for match in re.finditer(p, text, re.IGNORECASE):
            min_val, max_val = float(match.group(1)), float(match.group(2))
            results.append({'min': min_val, 'max': max_val, 'unit': unit, 'source_text': match.group(0)})
    return results


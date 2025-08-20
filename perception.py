from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from google import genai
import re
from typing import Dict

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Lightweight intent patterns (regex)
INTENT_PATTERNS: Dict[str, list[str]] = {
    'numerical_range': [
        r'(多少度|几度|温度|保持在|范围).*[°度CF]?',
        r'\d+\.?\d*\s*[-~至到]\s*\d+\.?\d*\s*[度°]?[CF]?',
        r'(几个月|多大|奶量|毫升|克|斤|kg|几次|几天)'
    ],
    'advice': [r'怎么办|如何|怎样|方法|建议|处理|解决|哄|安抚'],
    'factoid': [r'是什么|什么时候|多久|哪个|会不会|能不能|是否'],
    'definition': [r'什么是|是什么意思|定义|称作|称为'],
    'symptom_check': [r'正常吗|怎么回事|为什么|什么原因|什么病|严重吗|要不要去医院'],
    'product_recommendation': [r'推荐|哪个牌子|什么牌子|哪种好|买什么'],
    'irrelevant': [r'天气|新闻|股票|电影|游戏']
}

class PerceptionResult(BaseModel):
    user_input: str
    intent: Optional[str] = None
    entities: List[str] = []
    tool_hint: Optional[str] = None


def _rule_based_intent(text: str) -> Optional[str]:
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                return intent
    return None


def extract_perception(user_input: str) -> PerceptionResult:
    """Extracts intent, entities, and tool hints using rules with LLM fallback."""

    # 1) Rule-based intent first
    intent = _rule_based_intent(user_input)
    if intent:
        return PerceptionResult(user_input=user_input, intent=intent, entities=[], tool_hint=None)

    # 2) Fallback to LLM if rules don't catch it
    prompt = f"""
You are an AI that extracts structured facts from user input.

Input: "{user_input}"

Return the response as a Python dictionary with keys:
- intent: (brief phrase about what the user wants)
- entities: a list of strings representing keywords or values (e.g., ["INDIA", "ASCII"])
- tool_hint: (name of the MCP tool that might be useful, if any) Do not return null or empty string.

Output only the dictionary on a single line. Do NOT wrap it in ```json or other formatting. Ensure `entities` is a list of strings, not a dictionary.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        clean = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = eval(clean)
        except Exception as e:
            log("perception", f"Failed to parse cleaned output: {e}")
            raise
        if isinstance(parsed.get("entities"), dict):
            parsed["entities"] = list(parsed["entities"].values())
        return PerceptionResult(user_input=user_input, **parsed)
    except Exception as e:
        log("perception", f"Extraction failed: {e}")
        return PerceptionResult(user_input=user_input)

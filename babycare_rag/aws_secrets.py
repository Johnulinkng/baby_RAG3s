"""Helpers to fetch OpenAI API key from AWS Secrets Manager.

This is optional. If OPENAI_API_KEY is present in env, we do nothing.
If not, and SECRET_ID + AWS_REGION are set, we try to fetch the secret and
extract a likely OpenAI key from its JSON payload.
"""
from __future__ import annotations

import os
from typing import Optional


def get_openai_api_key_from_aws() -> Optional[str]:
    """Try to fetch OpenAI API key from AWS Secrets Manager.

    Looks for env:
      - SECRET_ID: the name/arn of the secret (e.g., Opean_AI_KEY_IOSAPP)
      - AWS_REGION: e.g., us-east-2

    Secret JSON may contain one of the keys below (checked in order):
      - OPENAI_API_KEY
      - OPENAI_IOS_KEY
      - openai_api_key
      - openai_key

    Returns:
      The API key string if found, else None.
    """
    secret_id = os.getenv("SECRET_ID")
    region = os.getenv("AWS_REGION")
    if not (secret_id and region):
        return None

    try:
        import json
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=secret_id)

        payload = resp.get("SecretString")
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            # If the secret is plain text (unlikely), just return it if it looks like a key
            if payload.startswith("sk-"):
                return payload
            return None

        # Try common key names
        for k in ("OPENAI_API_KEY", "OPENAI_IOS_KEY", "openai_api_key", "openai_key"):
            v = data.get(k)
            if isinstance(v, str) and v:
                return v
        return None
    except Exception:
        # Silently fail; caller will handle fallback
        return None


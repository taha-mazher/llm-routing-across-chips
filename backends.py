# One row per (model, chip). Add rows freely: everything else reads this list.
# usd_in / usd_out are list prices per million tokens (checked 2026-10-14). They feed
# the cost metric even though the free tiers charge nothing.
#
# Tried and dropped: Cerebras (free tier needs a card) and NVIDIA's trial API
# (retired gpt-oss-120b, and gpt-oss-20b took 3 to 60 seconds per answer), and
# gemini-3.8-flash (free tier allows 20 requests, then a long wait).
GROQ = dict(url="https://api.groq.com/openai/v1", key="GROQ_API_KEY")
GOOGLE = dict(url="https://generativelanguage.googleapis.com/v1beta/openai", key="GEMINI_API_KEY")

BACKENDS = [
    dict(name="gpt-oss-120b@groq-lpu", model="openai/gpt-oss-120b",
         usd_in=0.15, usd_out=0.60, extra={"reasoning_effort": "low"}, **GROQ),
    dict(name="gpt-oss-20b@groq-lpu", model="openai/gpt-oss-20b",
         usd_in=0.075, usd_out=0.30, extra={"reasoning_effort": "low"}, **GROQ),
    dict(name="gemini-3.1-flash-lite@google-tpu", model="gemini-3.1-flash-lite",
         usd_in=0.25, usd_out=1.50, **GOOGLE),
]

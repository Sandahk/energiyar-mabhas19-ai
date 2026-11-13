"""LLM suggest-and-confirm normalizer: one finite shadow experiment (not in the live path).

The deterministic engine answers first. Only when it fails closed (prose, refusal or out of
scope) may a model propose a structured reading of the question, chosen from closed printed
candidate sets; every field is verified deterministically, unverifiable fields are discarded,
and a valid proposal would only ever be shown as a confirmation question whose answer comes
from an unchanged engine call. Nothing here is imported by the API.
"""

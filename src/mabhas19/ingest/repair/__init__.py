"""Deterministic, evidence-based repair of the main PDF's damaged text layer.

Each operation is independent, switchable, and justified by glyph-level evidence
(font glyph ids, glyph geometry, rendered ink) rather than by dictionary guessing.
"""

"""Jev / Jev-like decision methods, one independent package per principle.

Per JEV_JA_LAB_REFACTOR_GUIDE_v2 section 4: method internals are not forced
into a shared inference interface. Each subpackage owns its own prompt
format (or lack thereof), forward pass, and score interpretation. What is
shared lives in `openjev_ja.common` (result schema, dataset identity) and
`openjev_ja.aggregate` (cross-method comparison).
"""

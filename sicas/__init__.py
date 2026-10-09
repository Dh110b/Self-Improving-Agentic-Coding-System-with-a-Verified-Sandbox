"""SICAS: Self-Improving Coding Agent with a verified Sandbox.

The agent improves its *scaffold* (prompts, tools, skills, retrieval, memory) - never model
weights. Every proposed scaffold change must pass a verification gate run in an isolated sandbox.
"""
__version__ = "0.1.0"

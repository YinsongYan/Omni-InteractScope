"""HTTP clients for hosted models, importable from both the package and ``scripts/``.

``metrics/llm_judge.py`` keeps its own client on purpose (it is frozen and locked to the binary
Correct/Incorrect path). What lives here is the general-purpose kind: a client that dataset
construction *and* a model adapter both need, which rules out ``scripts/`` as a home — an adapter
under ``omni_eval.models`` cannot ``sys.path.insert`` its way into a sibling script directory.
"""

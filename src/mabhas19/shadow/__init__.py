"""SH0: the LLM semantic proposer in shadow mode (no authority over regulatory truth).

A language model may *propose* the semantic structure of a question - clause roles, how each
identity-candidate span is used, which stated value is a lookup constraint, a claim to verify or
background, the requested table/output/unit, a calculation intent, coordination members. It
never supplies a value, a record, a citation or a name: every identifier it may use comes from
the deterministic :class:`~mabhas19.shadow.context.CandidateContext`.

* :mod:`~mabhas19.shadow.context` - the candidate context (sent to the model; no regulatory
  value) and the deterministic facts (verifier only), both read from one production run;
* :mod:`~mabhas19.shadow.proposal` - the typed proposal schema and its strict parser;
* :mod:`~mabhas19.shadow.verifier` - field-by-field deterministic verification, independent
  of model quality (a proposal is never evidence);
* :mod:`~mabhas19.shadow.translate` - verified components -> a restriction-only shadow plan,
  run through the unchanged deterministic pipeline (admission, I1-I11, execution,
  certification) for a counterfactual that is logged only;
* :mod:`~mabhas19.shadow.replay` - record/replay keys and records (no provider is called).

Nothing here is imported by the production retrieval modules: production results are
byte-identical with or without this package (tested).
"""

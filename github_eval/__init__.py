"""
Deep-agent eval harness for the GitHub tools.

Standalone by design (plan section 14.1): it manages mutable state in a real
repository, cannot fully clean up after itself, and grades with a model rather
than substring matching. None of that fits the one-off scripts in test_deepagent/.

Module boundaries that matter:
  cases.py    holds data, not logic -- adding a case must not mean writing code
  fixtures.py creates and best-effort removes; it never asserts
  judge.py    never touches GitHub or the tool service, so it can be exercised
              offline against recorded transcripts
  run.py      owns the finally, so teardown happens whatever the outcome
"""

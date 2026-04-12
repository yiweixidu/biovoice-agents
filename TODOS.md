# TODOS

## Grant Mode — Post-v1 Work

### Eval test for synthesis prompt quality
**What:** pytest-based eval that runs the grant synthesis prompt against a known
virology query and checks: [CITE:PMID] markers present in output, minimum word
count per section, no hallucinated PMIDs outside ranked_results.

**Why:** The synthesis prompt is the highest-risk component. Prompt changes that
degrade citation tagging won't surface until a real postdoc sees wrong output.
This eval enables confident prompt iteration with a regression safety net.

**How:** `@pytest.mark.eval` — skipped in normal CI, run manually or in a
dedicated eval pipeline. Requires a live LLM call against a hardcoded query
("broadly neutralizing antibodies influenza hemagglutinin") with a known-good
output snapshot to diff against.

**Pros:** Catches prompt regressions before users see them.
**Cons:** Requires LLM call; needs a baseline snapshot to compare against.
**Depends on:** Grant mode working end-to-end first (v1 shipped + validated).

---

### ClinicalTrials.gov integration for translational grant mode
**What:** Wire `clinicaltrials_agent.py` into a `--grant-type translational`
flag on the `biovoice grant` subcommand. `clinicaltrials_agent.py` is already
implemented. The work is: GrantConfig for translational mode + data mapping
from ClinicalTrials FetchResult into GrantSection structure.

**Why:** Translational grants (R01s bridging bench-to-bedside) need clinical
trial context. The design doc deferred this pending confirmation from v1 users
that their grant type needs it. This is the natural v1b expansion once validated.

**How:** Add `--grant-type` option to `biovoice grant` subcommand. Default:
`basic-science` (current behavior). Option: `translational` (adds ClinicalTrials
agent to the gather, adds clinical context section to GrantConfig prompts).

**Pros:** Opens tool to translational researchers (much larger grant market).
**Cons:** ClinicalTrials results need format mapping into GrantSection.
**Depends on:** v1 basic science mode validated with 2-3 postdocs first.

# Per-turn AI-interviewer evaluation

`evaluation/interviewer_turn_evaluation.py` judges the AI interviewer
one turn at a time, in context, over the REAL captured interview.
Live test: `tests/ai_quality/test_interviewer_turn_evaluation.py`
(writes `artifacts/reports/interviewer_turn_evaluation.json` with the
exact prompt, the exact transcript data, per-turn results and the
Python aggregation). Offline contract tests:
`tests/evaluation/test_interviewer_turn_evaluation.py`.

## Data flow

```
E2E capture (tests/e2e/test_bot_responsiveness.py)
  LiveKit data channel  -> artifacts/transcripts/livekit_transcript_events.json
      agent-<id>     TR_ ... caption   -> role=assistant, source=livekit-agent-session (medium)
      candidate-<id> TR_ ... caption   -> role=user,      source=livekit-candidate-stt  (high)
                                          = the EMH agent's OWN STT of the candidate
  DOM collector         -> actual_transcript.json  (user turns = injected-audio = harness script)
  local Whisper         -> stt_local_transcript.json (stt-local, medium)

select_turn_capture()      first backend with interviewer turns (+ candidate context preferred)
filter_turns()             keep interviewer turns; candidate turns kept ONLY if
                           source not in FORBIDDEN_CANDIDATE_SOURCES ("injected-audio")
                           and confidence >= medium; rejected turns leave a
                           "(candidate response not captured)" boundary
build_interviewer_turns()  one unit per interviewer utterance + verbatim preceding context
build_turn_prompt()        ONE prompt, every turn judged independently
nemotron_judge()           NVIDIA Nemotron, temperature 0, json_schema enforced
validate_turn_result()     count/numbering/score range/issue types
aggregate_scores()         mean of MODEL scores, issue counts, flagged turns
```

## Guarantees

* Candidate text the harness itself spoke (now the CandidateSimulator's
  `intended_text`; formerly the legacy `CANDIDATE_ANSWERS` script, kept
  only in `tests/fixtures/legacy_candidate_answers.py`) is tagged
  `injected-audio` by the collector and is rejected before anything
  reaches the judge (`test_prompt_is_invariant_to_candidate_answers_fixture`
  mutates/removes the legacy list and shows the prompt and results are
  identical).
* The judge never receives a predefined score; scores/issues in the
  report come from the model output and are only validated and
  averaged in Python.
* A candidate reply that was not captured is shown as
  `(candidate response not captured)`; the prompt forbids guessing it.

## Running

```
pytest tests/e2e/test_bot_responsiveness.py -s        # capture
pytest tests/ai_quality/test_interviewer_turn_evaluation.py -s
```

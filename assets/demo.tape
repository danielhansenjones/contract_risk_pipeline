# Verity README demo. Regenerate assets/demo.gif with:
#
#   scripts/secret-run docker compose up -d    # stack with /ask enabled, auth off
#   vhs assets/demo.tape                       # run from the repo root
#
# Requires vhs >= 0.8 (for Wait), ttyd, ffmpeg. Hidden blocks capture the job
# id via the idempotent replay, wait out the pipeline off-camera, and pre-warm
# the embedding model so the visible beats stay tight.

Output assets/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 16
Set TypingSpeed 30ms
Set Framerate 20

# Replace the prompt inherited from ~/.bashrc: keeps username/hostname out of
# every frame and gives Wait its expected "> " line ending to match.
Hide
Type "PROMPT_COMMAND=; PS1='> '"
Enter
Sleep 500ms
# Unique idempotency key per recording: without it the upload dedupes against
# a prior run's job and the whole demo replays already-completed state.
Type `DEMO_KEY="demo-$(date +%s)"`
Enter
Sleep 500ms
Type "clear"
Enter
Sleep 500ms
Show

# Beat 1: upload a contract
Type `curl -s -X POST localhost:8000/jobs -H "Idempotency-Key: $DEMO_KEY" -F file=@tests/test_documents/ServicesAgreement.pdf | jq`
Enter
Wait@30s
Sleep 3.5s

# Same key replays through the dedup path and returns the same job_id
Hide
Type `JOB=$(curl -s -X POST localhost:8000/jobs -H "Idempotency-Key: $DEMO_KEY" -F file=@tests/test_documents/ServicesAgreement.pdf | jq -r .job_id)`
Enter
Wait@30s
Type "clear"
Enter
Show

# Beat 2: the job is moving through pipeline stages
Type `curl -s localhost:8000/jobs/$JOB | jq '{status, stage}'`
Enter
Wait@30s
Sleep 3.5s

# Wait out classification/scoring off-camera; the services agreement is 17
# pages of CPU classification, so bound at 8 minutes
Hide
Type `for i in $(seq 1 240); do [ "$(curl -s localhost:8000/jobs/$JOB | jq -r .status)" = "completed" ] && break; sleep 2; done`
Enter
Wait@500s
Type "clear"
Enter
Show

# Beat 3: same poll, now completed
Type `curl -s localhost:8000/jobs/$JOB | jq '{status, stage}'`
Enter
Wait@30s
Sleep 3.5s

# Pre-warm the embedding model so the visible /ask answers quickly
Hide
Type `curl -s -X POST localhost:8000/jobs/$JOB/ask -H 'content-type: application/json' -d '{"question": "warmup", "top_k": 2}' > /dev/null`
Enter
Wait@120s
Type "clear"
Enter
Show

# Beat 4: the risk report - clause classification plus rule flags with
# verbatim matched text
Type `curl -s localhost:8000/jobs/$JOB/report | jq '{overall_score, risk_level, clause_summary, flags: [.flags[:3][] | {rule_id, severity, matched_text}]}'`
Enter
Wait@30s
Sleep 7s

Hide
Type "clear"
Enter
Show

# Beat 5: grounded /ask - the answer arrives with a verbatim citation
Type `curl -s -X POST localhost:8000/jobs/$JOB/ask -H 'content-type: application/json' -d '{"question": "What is the governing law of this agreement?"}' | jq '{answer, citations: [.citations[] | {quote, chunk_index}]}'`
Enter
Wait@120s
Sleep 6s
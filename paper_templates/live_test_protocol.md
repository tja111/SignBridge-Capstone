# SignBridge Live-Camera Accuracy Test Protocol

Use this protocol for the real-world evaluation section of the capstone paper.

## Purpose

Measure whether SignBridge recognizes supported signs correctly during live webcam use. This is different from the offline test-set accuracy because it includes real camera conditions, different signers, lighting, background, and hand-position variation.

## Recommended procedure

1. Test at least 2–3 participants. Obtain consent before collecting any data.
2. Use the same camera and record its model/resolution in the session notes.
3. Test Alphabet Mode and Words Mode separately.
4. For each supported sign, request 20 attempts per participant.
5. For each attempt, record one outcome: `Correct`, `Wrong`, `No recognition`, or `False positive`.
6. Include a no-sign control: record 20 camera frames/attempts with no supported sign visible. Any recognized sign is a false positive.
7. Do not alter the model or threshold during a test session.

## Conditions to record

- Participant ID (do not use a full name in the paper)
- Mode: Alphabet or Words
- Sign class
- Trial number
- Lighting: good / moderate / low
- Background: plain / cluttered
- Camera distance: near / normal / far
- Outcome
- Notes (for example: hand partly outside frame)

## Live-test summary table

Fill one row per participant and sign class. The file `live_test_summary_template.csv` is the spreadsheet version.

| Participant | Mode | Sign class | Attempts | Correct | Wrong | No recognition | False positives | Accuracy (%) | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| P1 | Words | Hello | 20 |  |  |  |  |  |  |
| P1 | Words | My Name Is | 20 |  |  |  |  |  |  |
| P1 | Words | Nice to Meet You | 20 |  |  |  |  |  |  |
| P1 | Words | Are You Alright? | 20 |  |  |  |  |  |  |
| P1 | Words | I am Fine | 20 |  |  |  |  |  |  |
| P1 | Words | I am Thirsty | 20 |  |  |  |  |  |  |
| P1 | Words | Wait | 20 |  |  |  |  |  |  |
| P1 | Words | Thank You | 20 |  |  |  |  |  |  |

Calculate accuracy as:

`Accuracy (%) = Correct / Attempts × 100`

Report false positives separately. A false positive is especially important when no supported sign is intentionally shown.

## Safe paper wording

> Live-camera evaluation was conducted using [number] participants under documented lighting, background, and distance conditions. Each supported sign was attempted [number] times. Outcomes were recorded as correct recognition, incorrect recognition, no recognition, or false positive. The live-camera results are reported separately from the held-out dataset evaluation.

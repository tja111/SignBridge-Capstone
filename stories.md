# SignBridge Capstone Paper Assistant Brief

Use this file as background context when asking ChatGPT to help write, revise, or organize the SignBridge capstone paper.

## Copy this instruction into ChatGPT first

You are assisting with a capstone paper about **SignBridge**, a Windows desktop prototype for real-time static sign recognition. Use only the verified project facts in this document. Do not invent accuracy, precision, recall, participant results, survey results, training duration, or claims of real-world translation quality. If a result is needed but is not supplied, write a clearly marked placeholder such as `[insert measured accuracy]` and tell the researcher what must be measured.

Use formal but clear academic writing. Describe SignBridge as a **prototype** and a **static sign-recognition system**, not as a complete sign-language translator. Distinguish between the trained system, measured evidence, and proposed future work.

## Project identity

| Item | Verified detail |
|---|---|
| Project name | SignBridge |
| Repository/model project name | SignDETR |
| Platform | Windows desktop application |
| Main purpose | Convert supported static hand signs from a webcam into recognized text; provide text-to-speech and text-to-sign playback support |
| Recognition approach | Custom DETR-style object detector implemented with PyTorch |
| Primary input | Live webcam frames |
| Primary output | Detected letters or supported words displayed as text |
| Additional output | Optional text-to-speech, saved transcript, copied text, sign viewer for typed text |
| Intended scope | Proof-of-concept assistive communication and research prototype |

## Important scope statement

SignBridge does **not** translate unrestricted sign language conversation. It recognizes a limited set of trained, static hand signs from individual webcam frames. The supported Words Mode items are treated as static classes for this project; they are not modeled as dynamic sign sequences.

## Recognition modes

### Alphabet Mode

- Supports 26 alphabet classes: A–Z.
- A detected letter must be held for approximately **0.8 seconds** before it is saved.
- The signer briefly releases or changes the hand position between letters.
- A pause of approximately **1.2 seconds** finalizes the current spelled word.
- The user can use Undo Letter/Backspace to correct a letter before or after a word is finalized.

### Words Mode

- Supports 8 trained classes.
- The same word sign must remain visible for approximately **2.5 seconds** before it is saved.
- This hold requirement reduces repeated and accidental text entries.
- The user must release or change the sign before saving the same word again.

## Active Words Mode classes and label IDs

The order below is the required numeric class mapping for the dataset, model, and application.

| ID | Class name | Display meaning |
|---:|---|---|
| 0 | `Hello` | Hello |
| 1 | `My_Name_IS` | My Name Is |
| 2 | `Nice_to_Meet_You` | Nice to Meet You |
| 3 | `Are_You_Alright?` | Are You Alright? |
| 4 | `I_am_Fine` | I am Fine |
| 5 | `I_am_Thirsty` | I am Thirsty |
| 6 | `Wait` | Wait |
| 7 | `ThankYou` | Thank You |

## Verified Words Mode dataset summary

Dataset counts below were checked from the current YOLO label files on September 2, 2026.

| Class | Train labels | Test labels | Total |
|---|---:|---:|---:|
| Hello | 307 | 64 | 371 |
| My Name Is | 297 | 65 | 362 |
| Nice to Meet You | 405 | 75 | 480 |
| Are You Alright? | 299 | 55 | 354 |
| I am Fine | 306 | 60 | 366 |
| I am Thirsty | 300 | 55 | 355 |
| Wait | 300 | 59 | 359 |
| Thank You | 208 | 43 | 251 |
| **Total** | **2,422** | **476** | **2,898** |

The dataset is imbalanced. `ThankYou` has the fewest examples and `Nice_to_Meet_You` has the most. The current Words Mode training code uses class-balanced sampling to reduce the effect of this imbalance.

## Model and training facts

- Framework: PyTorch and torchvision.
- Detection architecture: custom DETR-style detector with a ResNet-50 backbone.
- Bounding-box matching/training loss uses Hungarian matching.
- Data annotations use YOLO-style bounding boxes and numeric class IDs.
- Word training uses 320 × 320 input images.
- Words Mode default training profile: batch size 8; up to 400 epochs; early stopping patience 25; AdamW optimizer; cosine warm restarts; gradient clipping; GPU use when available.
- Class-balanced sampling is used for Words Mode training.
- The application automatically uses CUDA when it is available and otherwise runs on CPU.
- The Alphabet model and Words model are separate so word-model changes do not alter Alphabet Mode behavior.

## Application features

- Camera selection, Start/Stop, and Reset Camera controls.
- Alphabet/Words mode toggle.
- Live detected-sign, confidence, hold-progress, status, and last-saved indicators.
- Recognized sentence display with copy, save, clear, and undo controls.
- Optional text-to-speech, selected voice, speech rate, and volume controls.
- Typed text can be converted into a sign viewer for supported signs.
- Full-screen camera-only preview for demonstrations.
- Help popup, local-processing/privacy notice, saved local preferences, and diagnostic/log tools.
- A portable Windows build can be shared as a full application folder; the complete folder is required because it contains the executable, models, and runtime dependencies.

## Privacy and ethics wording

Use cautious wording such as:

> SignBridge processes webcam frames locally during use. The application does not intentionally save camera frames as part of the recognition workflow; however, users should still obtain consent before recording, testing, or collecting image data from participants.

Avoid claiming that the system replaces professional interpreters, provides medical or emergency guidance, or is suitable for unrestricted real-world sign-language translation.

## Suggested capstone chapter outline

1. Introduction
   - Communication accessibility problem
   - Project purpose and scope
   - Objectives and research questions
   - Significance and limitations

2. Review of Related Literature
   - Sign-language recognition
   - Computer vision and object detection
   - Transformer-based detection / DETR
   - Assistive communication technologies

3. Methodology
   - Development approach
   - Dataset collection and YOLO annotation
   - Class definitions and train/test split
   - DETR model, training settings, and deployment environment
   - Evaluation plan and ethical considerations

4. System Design and Implementation
   - Architecture: webcam → preprocessing → model → smoothing/hold logic → text/TTS/UI
   - Alphabet Mode and Words Mode workflows
   - Desktop interface and portable deployment

5. Results and Discussion
   - Insert measured recognition metrics
   - Present per-class confusion matrix if available
   - Discuss false positives, class imbalance, lighting, background, camera distance, and person-to-person variation

6. Conclusion and Recommendations
   - Summarize verified accomplishments
   - State constraints honestly
   - Recommend additional data, dynamic-sign modeling, broader evaluation, and user testing

## Metrics that must be measured before claiming results

Do not guess these values. Collect or provide them before writing the Results chapter.

- Overall detection/classification accuracy on a properly held-out test set.
- Per-class precision, recall, and F1 score.
- Confusion matrix for the eight Words Mode classes.
- False-positive rate when no supported sign is visible.
- Average recognition response time.
- Number of participants, test conditions, and device/camera settings.
- User evaluation or usability survey results, if any.

## Useful prompts for ChatGPT

### Write an introduction

Using the SignBridge Capstone Paper Assistant Brief, draft a 700-word Introduction with background, problem statement, objectives, scope, limitations, and significance. Do not invent performance results. Use placeholders where measured values are required.

### Write methodology

Using the brief, write the Methodology chapter in formal academic style. Explain the YOLO annotations, static image classes, train/test data split, custom DETR-style detector, class-balanced sampling, and desktop deployment. Clearly state that the system recognizes a limited set of static signs.

### Explain the architecture

Create a clear narrative for this pipeline: Webcam → frame preprocessing → DETR-style model → confidence/hold validation → recognized text → optional speech/transcript. Explain the separate Alphabet and Words modes in plain language.

### Draft results safely

Create a Results and Discussion chapter template for SignBridge. Use tables with placeholders for overall accuracy, per-class precision/recall/F1, false-positive rate, response time, and user feedback. Include a paragraph explaining how to interpret a confusion matrix.

### Improve writing

Rewrite the following paragraph for a capstone paper. Preserve factual meaning, use cautious academic language, do not add unsupported claims, and flag any sentence that requires a source or measured result: [paste paragraph]

## Facts to verify or fill in before final submission

- Final project title and institution format.
- Researchers' names, adviser, school, and academic year.
- Exact Alphabet dataset counts.
- Final Words Model training duration and hardware used.
- Final evaluation metrics from a truly held-out dataset.
- Participant count and consent procedure, if user testing is conducted.
- References for every claim made in the Review of Related Literature.

## Recommended language for limitations

> The system is limited to the signs represented in its training data and is sensitive to image quality, lighting conditions, background clutter, hand placement, camera position, and variation among signers. Because the prototype uses static-frame recognition, it does not model the motion and temporal grammar required for continuous sign-language translation.


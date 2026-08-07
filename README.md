

# IMAGE-QUALITY-ANALYZER
 An image quality assistant tool that combines classical Computer Vision techniques with Machine Learning to accurately evaluate image sharpness.

## Project Vision

### Build a hybrid ML-powered image quality analyzer that evaluates photographs in a way that is understandable and useful to photographers, providing objective measurements, intelligent explanations, and actionable recommendations for improving image quality.


## Architecture


```text
                              Input Image
                                  │
                                  ▼
                  ┌────────────────────────────────┐
                  │      Feature Extraction        │
                  │   (analyzer/feature_extractor) │
                  └────────────────────────────────┘
                                  │
        ┌──────────┬──────────┬───┴────┬──────────┬──────────┐
        ▼          ▼          ▼        ▼          ▼          ▼
   Laplacian    FFT Ratio  Wavelet  Noise RMS  Brightness  Contrast/
                            Ratio   (ml/noise)  /Clipping  Saturation/
                                                           Temperature
        │          │          │        │          │          │
        └──────────┴──────────┴───┬────┴──────────┴──────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  Sharp/Blurry Classifier  │
                    │   (Random Forest, 8 feat) │
                    └───────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │      Face Detection       │
                    │   (SCRFD, 106 landmarks)  │
                    └───────────────────────────┘
                                   │
                         for each face detected:
                                   ▼
              ┌────────────────────────────────────────┐
              │            Eye-State Detection         │
              │      (ML classifier + EAR fallback)    │
              │       Open / Closed / Left Closed /    │
              │       Right Closed / Undecided         │
              └────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │   Scoring & Grading       │
                    │   (analyzer/score.py)     │
                    │ Sharpness 40% + Noise 25% │
                    │ + Contrast 20% + Exp. 15% │
                    └───────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │    Report Generation       │
                    │ (analyzer/report_generator)│
                    │  grades, narrative text,   │──────────────────────────┐
                    │  JSON export, annotated    │                          │
                    │  image with overlays       │                          │
                    └────────────────────────────┘                          │
                                   │                                        │
                                   ▼                                        ▼
                    ┌───────────────────────────┐            ┌────────────────────────────────┐                 
                    │        Web Display        │            │  Saved into Logs               │
                    │   (Flask + index.html)    │            │  (Simplified and Comprehensive)│
                    └───────────────────────────┘            └────────────────────────────────┘ 
                                   │
                          face flagged as
                       Undecided or Closed?
                                   │YES
                                   ▼
                    ┌───────────────────────────┐
                    │    Human Review Loop      │
                    │  Reviewer marks eye state │
                    │  as intentional / not —   │
                    │  score is retroactively   │
                    │  updated (-15 penalty     │
                    │  only applied here, never │
                    │  automatically)           │
                    └───────────────────────────┘ 
```


## Project Structure

### Sharpness & Image Quality

`Laplacian`: Global edge intensity: How much neighbouring pixels disagree. Low = Smooth/Blurry, High = Sharp.

` FFT_Ratio`: Fraction of image's energy sitting in high frequencies(fine detail) vs broad shapes.

` Wavelet_Ratio`: Same idea as FFT but localized; captures where detail is, not just how much.

` Sharp_Ratio`: Fraction of 32x32 patches that individually counts as "Sharp".

` Consistency`: Whether sharpness is spread evenly across the image or concentrated in one small area.

` Noise`: Residual after Gaussian-blurring the image and subtracting from the original.

` Exposure`: A parabola centered on 50% brightness, and penalizes both under and over exposure.

` Detail Quality`: `fft_ratio/noise`; a simple detail to noise ratio.

-> These 8 features feed into a Random Forest Classifier trained on 100 hand-labeled sharp/blurry images, which outputs a sharp/blurry prediction with a confidence score:

Confidence	Grade

≥95% Sharp	Excellent
≥85% Sharp	Good
≥70% Sharp	Fair
<70% Sharp	Poor
<85% Blurry	Poor
≥85% Blurry	Very Poor

-> Noise, Contrast and Exposure are graded differently, scored against fixed thresholds in `analyzer/score.py`. The thresholds were adjusted after checking them against real processed images using the code found in `check_threshold.py`.

-> All 4 grades are then combined into one weighted score which is used as the main score for the image:

`overall_score = 0.40 × Sharpness + 0.25 × Noise + 0.20 × Contrast + 0.15 × Exposure`


### Face and Eye Detection

For face detection, the decision was made to use SCRFD (via InsightFace's buffalo_l model, which runs on CPU), replacing an earlier MediaPipe-based detectoer used in previous versions.

SCRFD returns, per face, in a single pass:
- A bounding box
- A detection confidence score
- 106 landmark points: Including 6 points per eye that everything in eye-state detection is built on. 

-> Getting both the box and landmarks from one model call(rather than incorporating a second landmark model) is why SCRFD replaced the older detector. Fewer moving parts, one source of truth per face, and easier to debug should something go wrong.

-> Each detected face is used to crop a thumbnail for the review UI. The crop is clamped to the image's actual dimensions before saving. SCRFD can return boxes that extend slightly past the frame edge for faces near the border, and cropping with an unclamped box can produce an empty image and crash the save step. A face whose box is entirely off-frame is skipped rather than crashing the whole request.

#### A little on SCRFD's landmarks
- SCRFD's 106-point scheme numbers: LEFT_EYE = [35,36,37,39,41,42] and RIGHT_EYE = [89,90,91,93,95,96] by image position (left side of the photo vs. right side), not by which eye anatomically belongs to the subject. 


For eye detection, each detected face gets classified into 1 of 4 states: Open, Closed, Left Closed, Right Closed, with a fifth non-model outcome, Undecided, for genuinely ambiguous cases.

Two independent methods can produce this classification, chosen per face:

- ML RandomForestClassifier: Default, used when the face is >= 160px and the model is at least 60% confident. The model is trained on 8 geometric features derived from 6 eye landmarks per eye:
-   -> left_ear/right_ear: Eye Aspect Ratio, the classical vertical-to-horizontal eyelid ratio.
-   -> avg_ear, eye_difference, ratio: derived comparisons between 2 eyes.
-   -> head_roll: head title angle, added to account for pose.
-   -> left_lid_bulge/right_lid_bulge: how far the top eyelid sits above the eye's own corner-to-corner line, measured relative to the eye's own geometry rather than the image's vertical axis. These two are the model's strongest features by a clear margin (importance 0.184 and 0.207 vs. 0.126–0.167 for everything else), and were the single biggest driver of the model's accuracy improvement over the session.
--- Add picture---

- EAR Fallback (used for small faces, or when the ML model's confidence is below 60%): A simpler, threshold-based method using only left_ear/right_ear against two cutoffs (a "closed" zone below 0.16, an "open" zone above 0.18). Genuinely ambiguous readings that fall in neither zone are reported as Undecided rather than guessed.

#### Eye Penalization

- The app does not automatically penalize a photo's score for closed eyes, in the event that the closed eye is intentional. Instead, any face whose eye state isn't a confirmed "Open" appears as a review card, where a human decides:

-   -> Intentional (e.g. a deliberate wink, or someone posing) — no penalty
-   -> Not intentional (a genuine accidental closed-eye moment) — 15 points, applied per face flagged this way

- Submitting a decision recomputes the score from scratch: `score = max(base_score - (15 x not_intentional_count), 0)`, then regenerates the verdict and report's narrative sections before saving the updated JSON.

- This design exists because eye-state detection, even at 92% cross-validated accuracy, will always have real edge cases(a wink, natural squint, small natural eyes etc) that a purely automatic system can't reliably tell apart from a genuine mistake. Rather than guessing and risking either false penalties or missed ones, the model flags only what it's unsure of, and a human makes the final call only on those specific cases, not on every photo.


### Report Generation

- Once scoring is complete, `analyzer/report_generator.py::generate_report_data()` assembles everything computed so far into one structured report. This is the single object that both the JSON export and the webpage render from, so nothing gets recomputed again.

The report has a few distinct parts:

`Quality`: Grades, scores, the weighted breakdown, and the human-review decision state (see The Review Workflow).

`technical_details`: The raw numeric values behind every metric (Laplacian, noise RMS, contrast, brightness, etc.), plus `technical_reference`, a parallel lookup of the actual threshold cutoffs behind whichever metrics have one, so a viewer isn't just shown a number with no sense of whether it's good or bad. So far the only metrics to have it are Noise, Contrast, Brightness, Shadow Clip and Highlight Clip.

`face_analysis`: Face count and per-face eye-state results, including which of Open/Closed/Left Closed/Right Closed/Undecided each one landed on.

`report`: The human-readable narrative: general (a plain summary line per metric), problems/suggestions (populated only when a grade is Fair or worse), strengths (populated only when a grade is Good or better), and summary (a few plain-English sentences describing the photo overall).

The finished report is written to disk twice, in two different forms:

`JSON/<filename>.json`: The full structured report, which is what actually gets re-read every time the web app displays or updates a report (nothing is recomputed on page load)

An annotated copy of the image `(annotated/visual_<filename>)`: Face boxes color-coded by eye state, plus an on-image dashboard overlay summarizing face/eye counts
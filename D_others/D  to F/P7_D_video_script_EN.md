# P7 · D — Video script (about 1 minute)

**Format:** screen recording with voice-over (PowerPoint "Record Slide Show", OBS or Zoom). No need to show your face.
**Length:** about 155 words at a normal pace (roughly 150 words per minute), so around 60–65 seconds.

| Time | On screen | Voice-over (read this) |
|---|---|---|
| 0:00–0:30 | Slide 1. Point at the top row while you describe the new-generator set (150 questions → 2 generators → same cleaning → 449 texts). Then point at the bottom row for the judge. Optional (last 3 seconds): open `P7_new_generators.csv` in a spreadsheet, or `P7_D_predict.py` scrolled to `SYSTEM_PROMPT`. | "My part has two deliverables. First, a new-generator test set: for 150 held-out HC3 questions, deepseek-chat and deepseek-reasoner wrote plain-text answers of about 100 to 160 words, cleaned exactly like the human answers, giving 449 texts for A, B, C and E to score. Second, an LLM judge: deepseek-flash at temperature zero scores each normalised text from 0 to 100, and I picked the few-shot prompt on the validation set only." |
| 0:30–0:47 | Slide 2. Point at 0.99 (HC3), 0.91 (RAID), 0.95 (new generators) and 0.76 (MAGE), then the conclusion bar. | "On HC3, AUROC is 0.99. It is 0.91 on a RAID subset, 0.95 on my new generators, and only 0.76 on MAGE. With a one-percent false-positive limit, detection on the new generators falls from 0.68 to 0.21." |
| 0:47–1:02 | Slide 3. Point at the blue "False positive" card, then the "0.12" row, then "Caveats". | "The judge’s reasons are informative, but it flags encyclopedic human text, twelve percent of human texts on MAGE, and it shares a vendor with the generators. Treat the score as a hint, not proof." |

**Before recording**
1. The "After attack" row on slide 2 still says TBD. Update it once you have `results/P7_D_attack.csv`.
2. Close notifications and make sure `.env` or any API key is not visible on screen.
3. Send the finished video to F.

**Sentence to add at the end of slide 2 once the attack results are in**
"After paraphrase attacks, AUROC goes from X to Y." (X and Y come from `results/P7_D_attack.csv`.)

# DigiSteel-YOLO: Automated Ablation Study Instructions

## Quick Start (Run While You Sleep)

### Option 1: One-Click Colab (Recommended)

1. Open this link in Chrome:
   ```
   https://colab.research.google.com/github/hazemelerefey/DigiSteel-YOLO/blob/main/notebooks/auto_ablation.ipynb
   ```

2. Enable GPU: **Runtime → Change runtime type → T4 GPU**

3. Click **Runtime → Run all** (or press Ctrl+F9)

4. **Go to sleep!** The notebook will run all experiments automatically (~8-10 hours)

5. When you wake up, check the results in `evals/` folder

---

### Option 2: Run Script in Colab

1. Open any Colab notebook

2. Run this single cell:
```python
!git clone https://github.com/hazemelerefey/DigiSteel-YOLO.git
%cd DigiSteel-YOLO
!bash scripts/run_ablation.sh
```

3. **Go to sleep!** (~8-10 hours)

---

## What Will Run Automatically

| Experiment | Duration | Expected Result |
|---|---|---|
| C3Ghost Architecture | ~1.5 hours | Better parameter efficiency |
| Image Size 800 | ~2 hours | Better small defect detection |
| Enhanced Augmentation | ~1.5 hours | Better generalization |
| Cosine Learning Rate | ~1.5 hours | Better convergence |
| **Final Optimized Model** | ~2.5 hours | **Best mAP@0.5** |

**Total: ~8-10 hours**

---

## Results You'll Get

After completion, you'll have:

1. **`evals/best_config.json`** — Best configuration found
2. **`evals/final_results.json`** — Final model performance
3. **`evals/ablation_study_final_report.md`** — Comprehensive report
4. **`runs/final_optimized_neu_det_seed42/weights/best.pt`** — Best model weights

---

## Goal

**Beat all 11 reference papers:**

| Paper | mAP@0.5 | Our Target |
|---|---|---|
| P10 KDM-YOLO | 95.4% | >95.4% |
| P02 LAM-YOLOv10n | 94.39% | >94.39% |
| P03 YOLO-LSDI | 83.0% | >83.0% |
| P07 ASFRW-YOLO | 83.2% | >83.2% |

---

## After Completion

1. Check `evals/ablation_study_final_report.md` for results
2. Review `evals/best_config.json` for optimal configuration
3. The final model is in `runs/final_optimized_neu_det_seed42/weights/best.pt`

---

**Good night! The automation will handle everything.** 🚀

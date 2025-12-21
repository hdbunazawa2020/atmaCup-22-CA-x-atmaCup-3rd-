# About this
- about this repository

## 📊 スコア推移

| Date | EXP      | DATA   | CV       | MODEL    | Ep  | CV    | LB     | CONTENT     | RESULT               | 
|------|----------|--------|----------|----------|-----|-------|--------|-------------|----------------------|
| 12/15 | exp001  | ver00  | hold_out | arcface  | 20  | 0.81  | 0.7753 | baseline    | CV-LBのシフト大       |
| 12/15 | exp002  | ver00  | hold_out | arcface  | 20  | 0.81  | 0.7281 | margin threshold追加 | 効果があまりみられず       |
| 12/17 | exp003  | ver01  | hold_out | arcface  | 20  | 0.81  | 0.8452 | 負例データ追加 | LBが良化した       |
| 12/17 | exp004  | ver01  | hold_out | arcface  | 20  | 0.81  | 0.8391 | classwise threshold | LBは微妙に悪化       |
| 12/18 | exp005  | ver01  | hold_out | arcface  | 20  | 0.814  | 0.8509 | 100_trainにCVも導入 | LB良化       |
| 12/20 | exp006  | ver02  | 4-fold_cv | arcface  | 20  | 0.814  | 0.8822 | 4-foldで学習 | LB良化       |
| 12/21 | exp008  | ver03  | 4-fold_cv | arcface  | 20  | 0.814  | 0.8795 | img_sizeを拡大(256->384) | 効果なし       |
| 12/21 | ens000  | -      | 4-fold_cv | arcface  | 20  | 0.814  | 0.8878 | exp006*0.55+exp008*0.45 | 効果あり       |
| 12/21 | exp010  | ver03  | 4-fold_cv | arcface  | 20  | 0.814  | 0.8--- | convnext_base | -       |
| 12/21 | exp011  | ver03  | 4-fold_cv | arcface  | 20  | 0.814  | 0.8584/0.8707(withILP) | efficientnet_b1 | 精度下がった       |
| 12/21 | exp012  | ver03  | 4-fold_cv | arcface  | 20  | 0.814  | 0.8669 | efficientnet_b3 | 精度下がった       |

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

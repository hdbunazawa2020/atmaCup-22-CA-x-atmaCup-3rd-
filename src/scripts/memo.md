### 2GPUで並列処理する時
```py
torchrun --nproc_per_node=2 --master_port=12345 200_unet/200_unet.py
```
### nohupで2GPUで並列処理
```py
nohup torchrun --nproc_per_node=2 201_convnext/201_convnext.py > train.log 2>&1 &
```
### 実行の確認
```bash
tail -f nohup.out
```
### ログの表示
```bash
tail -f train.log
```
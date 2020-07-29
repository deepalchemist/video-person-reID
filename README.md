# Video-Person-reID

### Training & Evaluation

- Training

```
python main.py -d mars \
  --seq-len 8 --height 256 --width 128 --train-batch 32 --test-batch 32 \
  --train-sample-method interval --test-sample-method evenly \
  --arch=tsn --pool-type max --non-local --stm cmm \
  --lr 3e-4 --warmup-epoch 10 --stepsize 40 --eval-steps 0 50 90 130 \
  --ckpt-dir 'dir-name-to-save-the-checkpoint' \
  --gpu-devices 0,1 \
```

- Evaluate

```
python main.py -d mars \
  --seq-len 8 --height 256 --width 128 --test-batch 32 \
  ----test-sample-method evenly \
  --arch=tsn --pool-type max --non-local --stm cmm \
  --resume 'absolute-path-to-the-checkpoint' \
  --gpu-devices 0,1 \
```
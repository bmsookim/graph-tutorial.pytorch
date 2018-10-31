python train.py \
    --dataset citeseer \
    --num_hidden 32 \
    --dropout 0.5 \
    --weight_decay 5e-4 \
    --model attention \
    --lr 1e-3 \
    --optimizer adam \
    --epoch 1000 \
    --lr_decay_epoch 250

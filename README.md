# Implementation of Deep Hybrid Models for Out-of-Distribution (OOD) detection


## Replicating CIFAR-10 Results

```commandline
python train_dhm_iresflow.py --epochs 200 --batch 256 --N 4 --k 10 --n_blocks 10 --dims 640 --vnorms 222222 --lamb 0.000375 --sn True --n_power_iter 1 --dnn_coeff 3.0 --lr_schedule 60-120-160 --test_every_epoch True --test_model True --seed 0 --lr 1e-4 --flatten True --normalise_features True --distribution_model normflow --actnorm False --save_checkpoints True --dirpath checkpoints/dhm-alt-dists
```


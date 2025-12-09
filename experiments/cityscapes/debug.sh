method=fairgrad
alpha=2.0
seed=42
cuda=0
lora_epoch=0
tele_sign=True
tele_times_per_epoch=1

CUDA_VISIBLE_DEVICES=$cuda python3 trainer.py --method=$method --tele_sign=$tele_sign --tele_times_per_epoch=$tele_times_per_epoch --lora_epoch=$lora_epoch --seed=$seed --alpha=$alpha

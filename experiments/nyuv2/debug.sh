method=fairgrad
alpha=2.0
seed=42
cuda=1
lora_epoch=0

CUDA_VISIBLE_DEVICES=$cuda python3 trainer.py --method=$method --lora_epoch=$lora_epoch --seed=$seed

mkdir -p ./save
mkdir -p ./trainlogs

method=fairgrad
alpha=2.0
seed=44
cuda=3
lora_epoch=50
tele_sign=True
tele_times_per_epoch=1

# CUDA_VISIBLE_DEVICES=$cuda python3 trainer.py --method=$method --seed=$seed --alpha=$alpha > trainlogs/$method-alpha$alpha-sd$seed.log 2>&1 &
CUDA_VISIBLE_DEVICES=$cuda python3 trainer.py --method=$method --lora_epoch=$lora_epoch --tele_sign=$tele_sign --tele_times_per_epoch=$tele_times_per_epoch --seed=$seed --alpha=$alpha > trainlogs/$method-lora_epoch$lora_epoch-alpha$alpha-sd$seed.log 2>&1 &

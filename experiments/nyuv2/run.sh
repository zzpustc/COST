mkdir -p ./save
mkdir -p ./trainlogs

method=fairgrad
alpha=2.0
seed=44
cuda=3
lora_epoch=50

CUDA_VISIBLE_DEVICES=$cuda python3 trainer.py --method=$method --seed=$seed --lora_epoch=$lora_epoch --alpha=$alpha > trainlogs/$method-lora_epoch$lora_epoch-alpha$alpha-sd$seed.log 2>&1 &

export SQUAD_DIR=/capstone/SQuAD_v1.1
export PYTORCH_TRANSFORMERS_CACHE=/capstone/squad_cache/
python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad\
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /capstone/SQuAD_v1.1/out

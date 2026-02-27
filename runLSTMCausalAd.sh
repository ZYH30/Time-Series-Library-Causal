# 文件路径：Time-Series-Library/runLSTMCausalAd.sh
#!/bin/bash
# LSTMCausalAd Fair Benchmarking Execution Script (Route 2)
# 采用 MS (多变量输入，单变量预测)
# 通过 --past_features 传入因果预筛选的变量列表

export CUDA_VISIBLE_DEVICES=0

# 你的因果特征集合
PAST_FEATURES="wd_deg SWDR_W max_wv wv_m rho_g max_PAR VPdef_mbar PAR_ol VPmax_mbar rh Tpot_K"

python -u runLSTMCausalAd.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_$pred_len_causal \
  --model LSTMCausalAd \
  --features MS \
  --target OT \
  --past_features $PAST_FEATURES \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --hidden_size 4 \
  --num_layers 2 \
  --embedding_size 12 \
  --hidden_size_target 4 \
  --batch_size 1024 \
  --learning_rate 0.015892 \
  --lr_adv 0.005 \
  --adv_weight 0.001 \
  --train_epochs 36 \
  --normal_epochs 5 \
  --adv_epochs 5 \
  --patience 20 \
  --dropout 0.2
  # ⚠️ 注意：这里刻意删除了 --inverse，以输出约为 0.5 水平的标准化 MSE
  
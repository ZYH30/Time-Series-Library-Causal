#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PAST_FEATURES="wd_deg SWDR_W max_wv wv_m rho_g max_PAR VPdef_mbar PAR_ol VPmax_mbar rh Tpot_K"
FORW_FEATURES=""
# FORW_FEATURES="year month day"

for pred_len in 96 192 336 720
do
    echo "=========================================================="
    echo "Running Causal Attention Adversarial Evaluation for Horizon: $pred_len"
    echo "=========================================================="
    
    python -u runLSTMCausalPostAttAd.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_$pred_len_causal \
        --model LSTMCausalPostAttentionAd \
        --features MS \
        --target OT \
        --past_features $PAST_FEATURES \
        --forward_features $FORW_FEATURES \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --hidden_size 16 \
        --attn_head 4 \
        --attn_head_target 8 \
        --num_layers 2 \
        --embedding_size 8 \
        --hidden_size_target 16 \
        --batch_size 1024 \
        --learning_rate 0.5 \
        --lr_adv 0.01 \
        --adv_weight 2.4e-05 \
        --train_epochs 64 \
        --normal_epochs 9 \
        --adv_epochs 3 \
        --patience 10 \
        --dropout 0.2
done

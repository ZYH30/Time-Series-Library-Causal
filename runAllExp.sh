#!/bin/bash
# LSTMCausalPostAttentionAd Fair Benchmarking Execution Script 

export CUDA_VISIBLE_DEVICES=0

run_timed() {
    local command_name="$1"
    local start_time
    local end_time
    local duration
    
    shift
    echo "=========================================================="
    echo "[Starting] $command_name"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================================="
    
    start_time=$(date +%s)
    
    echo "cmd: $@"
    echo "----------------------------------------------------------"
    "$@"
    local exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo "=========================================================="
    echo "[Endding] $command_name"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ $hours -gt 0 ]; then
        echo "Running time: ${hours} h ${minutes} m ${seconds} s (Total: ${duration} seconds.)"
    elif [ $minutes -gt 0 ]; then
        echo "Running time: ${minutes} m ${seconds} s (总计: ${duration}秒)"
    else
        echo "Running time: ${duration} seconds."
    fi
    
    echo "Exit: ${exit_code}"
    echo "=========================================================="
    echo ""
    
    return $exit_code
}

echo "=========================================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

model_name=LSTMCausalPostAttentionAd

# causal features
# PAST_FEATURES="p_mbar Tdew_degC VPact_mbar H2OC_m sh_g SWDR_W PAR_ol max_PAR rho_g Tlog_degC VPmax_mbar T_degC Tpot_K wd_deg max_wv wv_m VPdef_mbar rh rain_mm raining_s"
PAST_FEATURES="wd_deg SWDR_W max_wv wv_m rho_g max_PAR VPdef_mbar PAR_ol VPmax_mbar rh Tpot_K"
# FORW_FEATURES="year month day"
FORW_FEATURES=""

pred_len=96

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u runLSTMCausalPostAttAd.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id causal_weather_96_$pred_len \
  --model LSTMCausalPostAttentionAd \
  --features MS \
  --target OT \
  --past_features $PAST_FEATURES \
  --forward_features $FORW_FEATURES\
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

model_name=LSTMCausalAd

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u runLSTMCausalAd.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id causal_weather_96_$pred_len \
  --model LSTMCausalAd \
  --features MS \
  --target OT \
  --past_features $PAST_FEATURES \
  --forward_features $FORW_FEATURES \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
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
  --patience 10 \
  --dropout 0.2
      
model_name=Autoformer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 2

model_name=Crossformer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

model_name=FiLM

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model FiLM \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1

model_name=iTransformer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \

model_name=MICN

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

model_name=MultiPatchFormer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1

model_name=Nonstationary_Transformer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2

model_name=PatchTST

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --train_epochs 3

model_name=Pyraformer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 2

model_name=SegRNN

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

seq_len = 96

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len 48 \
  --enc_in 21 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1

model_name=TimeMixer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

model_name=TimesNet

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

model_name=TimeXer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1 \

model_name=Transformer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3

model_name=TSMixer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

model_name=WPMixer

echo "=========================================================="
echo "Running $model_name for Horizon: $pred_len"
echo "=========================================================="

run_timed "$model_name" \
python -u run.py \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model_id wpmixer \
    --model $model_name \
    --task_name long_term_forecast \
    --data custom \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --label_len 0 \
    --d_model 256 \
    --patch_len 16 \
    --batch_size 32 \
    --learning_rate 0.000913333 \
    --lradj type3 \
    --dropout 0.4 \
    --patience 12 \
    --train_epochs 60 \
    --use_amp

echo "=========================================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

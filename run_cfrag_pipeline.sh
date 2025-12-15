#!/bin/bash
# CFRAG 完整訓練流程自動化腳本
# 解決所有路徑和 GPU 配置問題

set -e  # 遇到錯誤立即退出

# ==================== 配置區域 ====================
# 請根據你的環境修改以下配置

# GPU 設置
CUDA_DEVICE="2"  # 使用的 GPU 編號

# 任務配置
TASK="LaMP_2_time"
SOURCE="recency"
DATA_SPLIT_TRAIN="train"
DATA_SPLIT_DEV="dev"

# 模型配置
MODEL_NAME="Qwen2-7B-Instruct"
BASE_RETRIEVER_PATH="LLMs/bge-base-en-v1.5"
BASE_RERANKER_PATH="LLMs/bge-reranker-base"

# 路徑配置
PROJECT_ROOT="/mnt/data/ckt77/IR_final/CFRAG"
DATA_DIR="${PROJECT_ROOT}/data/${TASK}"
OUTPUT_DIR="${PROJECT_ROOT}/${MODEL_NAME}_outputs/${TASK}"

# 是否跳過已完成的步驟（用於斷點續跑）
SKIP_STEP1=false      # 跳過步驟1（User Embedding Training）
SKIP_STEP2=false 
SKIP_STEP2_1=false      # 跳過步驟2.1（獲取Retriever訓練數據），直接從2.2開始
SKIP_STEP3=false 
SKIP_STEP3_1=false     # 是否跳過步驟3.1（獲取ReRanker訓練數據）
SKIP_STEP4=false

# 相似用戶聚合模組配置
USE_SIMILAR_USER_AGGREGATION=false  # 是否啟用相似用戶聚合模組
SIMILAR_USER_TOPK=3                # 檢索的 top-k 相似用戶數量
AGGREGATION_METHOD="attention"    # 聚合方法: "attention" 或 "weighted_sum"
SIMILAR_USER_WEIGHT=0.3            # 相似用戶 embedding 的權重 (0.0-1.0)
USE_PREVIOUS_USER_EMB=true         # 如果存在之前的 user embeddings，是否使用它們作為 lookup

# ==================== 工具函數 ====================

# 打印帶顏色的消息
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# 檢查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        print_error "文件不存在: $1"
        return 1
    fi
    return 0
}

# 檢查目錄是否存在
check_dir() {
    if [ ! -d "$1" ]; then
        print_error "目錄不存在: $1"
        return 1
    fi
    return 0
}

# 獲取最新的用戶嵌入文件（只返回文件名）
get_latest_user_emb_file() {
    local user_emb_dir="${DATA_DIR}/${DATA_SPLIT_DEV}/${SOURCE}/user_emb"
    if [ ! -d "$user_emb_dir" ]; then
        print_error "用戶嵌入目錄不存在: $user_emb_dir"
        return 1
    fi
    local latest_file=$(ls -t "${user_emb_dir}"/*.pt 2>/dev/null | head -1)
    if [ -z "$latest_file" ]; then
        print_error "未找到用戶嵌入文件"
        return 1
    fi
    basename "$latest_file"
}

# 獲取最新的 retriever checkpoint（返回相對路徑）
get_latest_retriever_checkpoint() {
    local checkpoint_dir="${OUTPUT_DIR}/${DATA_SPLIT_TRAIN}/${SOURCE}"
    
    # 查找 bge-base-en-v1.5 目錄（不帶下劃線和數字）
    local model_base_dir="${checkpoint_dir}/bge-base-en-v1.5"
    if [ ! -d "$model_base_dir" ]; then
        print_error "未找到 Retriever checkpoint 基礎目錄: $model_base_dir" >&2
        return 1
    fi
    
    # 在基礎目錄下找最新的時間戳目錄
    local latest_timestamp_dir=$(ls -td "${model_base_dir}"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | head -1)
    if [ -z "$latest_timestamp_dir" ]; then
        print_error "未找到 Retriever checkpoint 時間戳目錄" >&2
        print_error "搜索目錄: $model_base_dir" >&2
        return 1
    fi
    
    # 返回格式: 模型名/時間戳（例如: bge-base-en-v1.5/20251212-110059）
    local model_name=$(basename "$model_base_dir")
    local timestamp=$(basename "$latest_timestamp_dir")
    echo "${model_name}/${timestamp}"
}

# 獲取最新的 reranker checkpoint（返回相對路徑）
get_latest_reranker_checkpoint() {
    local checkpoint_dir="${OUTPUT_DIR}/${DATA_SPLIT_TRAIN}/${SOURCE}"
    
    # 查找 bge-reranker-base 目錄
    local model_base_dir="${checkpoint_dir}/bge-reranker-base"
    if [ ! -d "$model_base_dir" ]; then
        print_error "未找到 ReRanker checkpoint 基礎目錄: $model_base_dir" >&2
        return 1
    fi
    
    # 在基礎目錄下找最新的時間戳目錄
    local latest_timestamp_dir=$(ls -td "${model_base_dir}"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | head -1)
    if [ -z "$latest_timestamp_dir" ]; then
        print_error "未找到 ReRanker checkpoint 時間戳目錄" >&2
        print_error "搜索目錄: $model_base_dir" >&2
        return 1
    fi
    
    # 返回格式: 模型名/時間戳（例如: bge-reranker-base/20251208-155113）
    local model_name=$(basename "$model_base_dir")
    local timestamp=$(basename "$latest_timestamp_dir")
    echo "${model_name}/${timestamp}"
}

# 獲取最新的檢索結果文件名（不含路徑和擴展名）
# 參數：可選的數據集名稱（train 或 dev），默認為 train
get_latest_retrieval_file_name() {
    local data_split=${1:-$DATA_SPLIT_TRAIN}  # 默認使用 train，可以傳入 dev
    local retrieval_dir="${OUTPUT_DIR}/${data_split}/${SOURCE}"
    
    # 找到所有 retrieval 目錄
    local retrieval_dirs=$(find "$retrieval_dir" -type d -name "retrieval" 2>/dev/null)
    
    if [ -z "$retrieval_dirs" ]; then
        print_error "未找到 retrieval 目錄" >&2
        print_error "搜索目錄: $retrieval_dir" >&2
        return 1
    fi
    
    # 在每個 retrieval 目錄中找最新的文件（按修改時間）
    local latest_file=""
    local latest_time=0
    
    for dir in $retrieval_dirs; do
        if [ -d "$dir" ]; then
            # 使用 ls -t 按時間排序，獲取最新的文件
            local file=$(ls -t "$dir"/*.json 2>/dev/null | head -1)
            if [ -n "$file" ] && [ -f "$file" ]; then
                local file_time=$(stat -c %Y "$file" 2>/dev/null || echo 0)
                if [ "$file_time" -gt "$latest_time" ]; then
                    latest_time=$file_time
                    latest_file="$file"
                fi
            fi
        fi
    done
    
    if [ -z "$latest_file" ] || [ ! -f "$latest_file" ]; then
        print_error "未找到檢索結果文件" >&2
        print_error "搜索目錄: $retrieval_dir" >&2
        return 1
    fi
    
    print_info "找到最新文件: $latest_file" >&2
    basename "$latest_file" .json
}

# 獲取最新的 rerank 結果文件名
get_latest_rerank_file_name() {
    local rerank_dir="${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}"
    
    # 找到所有 rerank 目錄或包含 rerank 的路徑
    local rerank_dirs=$(find "$rerank_dir" -type d -name "*rerank*" 2>/dev/null)
    
    if [ -z "$rerank_dirs" ]; then
        # 如果沒有找到 rerank 目錄，嘗試在 source 目錄下查找
        rerank_dirs=$(find "$rerank_dir" -type d 2>/dev/null | grep -i rerank)
    fi
    
    if [ -z "$rerank_dirs" ]; then
        print_error "未找到 rerank 目錄" >&2
        print_error "搜索目錄: $rerank_dir" >&2
        return 1
    fi
    
    # 在每個目錄中找最新的文件（按修改時間）
    local latest_file=""
    local latest_time=0
    
    for dir in $rerank_dirs; do
        if [ -d "$dir" ]; then
            # 使用 ls -t 按時間排序，獲取最新的文件
            local file=$(ls -t "$dir"/*.json 2>/dev/null | head -1)
            if [ -n "$file" ] && [ -f "$file" ]; then
                local file_time=$(stat -c %Y "$file" 2>/dev/null || echo 0)
                if [ "$file_time" -gt "$latest_time" ]; then
                    latest_time=$file_time
                    latest_file="$file"
                fi
            fi
        fi
    done
    
    # 如果還是沒找到，嘗試在整個目錄下查找
    if [ -z "$latest_file" ] || [ ! -f "$latest_file" ]; then
        local all_files=$(find "$rerank_dir" -name "*.json" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
        if [ -n "$all_files" ] && [ -f "$all_files" ]; then
            latest_file="$all_files"
        fi
    fi
    
    if [ -z "$latest_file" ] || [ ! -f "$latest_file" ]; then
        print_error "未找到 Rerank 結果文件" >&2
        print_error "搜索目錄: $rerank_dir" >&2
        return 1
    fi
    
    print_info "找到最新 Rerank 文件: $latest_file" >&2
    basename "$latest_file" .json
}

# ==================== 主流程 ====================

cd "$PROJECT_ROOT" || exit 1

print_info "開始 CFRAG 完整訓練流程"
print_info "項目根目錄: $PROJECT_ROOT"
print_info "使用 GPU: $CUDA_DEVICE"

# 設置環境變數（必須在導入 PyTorch 之前）
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ==================== 步驟 1: User Embedding Training ====================
if [ "$SKIP_STEP1" = false ]; then
    print_info "========== 步驟 1: User Embedding Training =========="
    
    # 1.1 數據預處理（如果需要的話，可以根據實際情況調整）
    print_info "1.1 數據預處理..."
    # python data/preprocess_profile.py --data_phase train
    # python data/preprocess_profile.py --data_phase dev
    # python user_emb/get_user_set.py
    # python user_emb/get_corpus_emb.py
    
    # 1.2 訓練用戶嵌入
    print_info "1.2 訓練用戶嵌入..."
    cd user_emb/train_user_emb
    
    # 構建訓練命令參數
    TRAIN_ARGS="--use_attention_pooling False"
    
    # 如果啟用相似用戶聚合模組
    if [ "$USE_SIMILAR_USER_AGGREGATION" = true ]; then
        print_info "啟用相似用戶聚合模組..."
        TRAIN_ARGS="$TRAIN_ARGS --use_similar_user_aggregation True"
        TRAIN_ARGS="$TRAIN_ARGS --similar_user_topk $SIMILAR_USER_TOPK"
        TRAIN_ARGS="$TRAIN_ARGS --aggregation_method $AGGREGATION_METHOD"
        TRAIN_ARGS="$TRAIN_ARGS --similar_user_weight $SIMILAR_USER_WEIGHT"
        
        # 如果啟用使用之前的 user embeddings 作為 lookup
        if [ "$USE_PREVIOUS_USER_EMB" = true ]; then
            # 檢查是否存在之前的 user embeddings
            PREV_USER_EMB_DIR="${DATA_DIR}/${DATA_SPLIT_DEV}/${SOURCE}/user_emb"
            if [ -d "$PREV_USER_EMB_DIR" ]; then
                PREV_USER_EMB_FILE=$(ls -t "${PREV_USER_EMB_DIR}"/*.pt 2>/dev/null | head -1)
                if [ -n "$PREV_USER_EMB_FILE" ] && [ -f "$PREV_USER_EMB_FILE" ]; then
                    # 構建相對於 train_user_emb 目錄的路徑
                    PREV_USER_EMB_REL_PATH="../../data/${TASK}/${DATA_SPLIT_DEV}/${SOURCE}/user_emb/$(basename "$PREV_USER_EMB_FILE")"
                    TRAIN_ARGS="$TRAIN_ARGS --user_emb_lookup_path $PREV_USER_EMB_REL_PATH"
                    print_info "使用之前的用戶嵌入作為 lookup: $(basename "$PREV_USER_EMB_FILE")"
                else
                    print_warning "未找到之前的用戶嵌入文件，將不使用 lookup path"
                fi
            else
                print_warning "用戶嵌入目錄不存在，將不使用 lookup path"
            fi
        fi
    else
        print_info "相似用戶聚合模組已關閉"
    fi
    
    # 執行訓練
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python run.py $TRAIN_ARGS
    cd "$PROJECT_ROOT"
    
    # 驗證用戶嵌入是否生成
    USER_EMB_FILE=$(get_latest_user_emb_file)
    if [ $? -eq 0 ]; then
        print_success "用戶嵌入訓練完成: $USER_EMB_FILE"
    else
        print_error "用戶嵌入訓練失敗"
        exit 1
    fi
else
    print_warning "跳過步驟 1，使用現有的用戶嵌入"
    USER_EMB_FILE=$(get_latest_user_emb_file)
    if [ $? -ne 0 ]; then
        print_error "未找到用戶嵌入文件"
        exit 1
    fi
fi

# ==================== 步驟 2: Retriever Training ====================
if [ "$SKIP_STEP2" = false ]; then
    print_info "========== 步驟 2: Retriever Training =========="
    
    # 2.1 獲取 Retriever 訓練數據
    if [ "$SKIP_STEP2_1" = false ]; then
        print_info "2.1 獲取 Retriever 訓練數據..."
        cd "$PROJECT_ROOT"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ranking.py \
            --rank_stage retrieval \
            --data_split $DATA_SPLIT_TRAIN \
            --ret_type dense \
            --base_retriever_path "$BASE_RETRIEVER_PATH" \
            --user_emb_path "$USER_EMB_FILE" \
            --CUDA_VISIBLE_DEVICES $CUDA_DEVICE
        
        # 獲取生成的文件名
        RETRIEVAL_FILE_NAME=$(get_latest_retrieval_file_name)
        if [ $? -ne 0 ]; then
            print_error "未找到檢索結果文件"
            exit 1
        fi
        print_success "檢索完成，文件名: $RETRIEVAL_FILE_NAME"
    else
        print_warning "跳過步驟 2.1，使用現有的檢索結果"
        cd "$PROJECT_ROOT"
        RETRIEVAL_FILE_NAME=$(get_latest_retrieval_file_name)
        if [ $? -ne 0 ]; then
            print_error "未找到檢索結果文件"
            exit 1
        fi
        print_success "使用現有檢索結果，文件名: $RETRIEVAL_FILE_NAME"
    fi
    
    # 2.2 生成訓練點
    print_info "2.2 生成訓練點..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generation/generate_point.py \
        --source retrieval \
        --file_name "$RETRIEVAL_FILE_NAME" \
        --CUDA_VISIBLE_DEVICES $CUDA_DEVICE
    
    # 2.3 Retriever 訓練
    print_info "2.3 Retriever 訓練..."
    cd rank_tune/retriever
    # 構建完整的用戶嵌入路徑（相對於 rank_tune/retriever 目錄）
    USER_EMB_FULL_PATH="../../data/${TASK}/${DATA_SPLIT_DEV}/${SOURCE}/user_emb/${USER_EMB_FILE}"
    print_info "使用用戶嵌入: $USER_EMB_FULL_PATH"
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python run.py \
        --user_emb_path "$USER_EMB_FULL_PATH"
    cd "$PROJECT_ROOT"
    
    # 驗證 checkpoint
    RETRIEVER_CKPT=$(get_latest_retriever_checkpoint)
    if [ $? -eq 0 ]; then
        print_success "Retriever 訓練完成: $RETRIEVER_CKPT"
    else
        print_error "Retriever 訓練失敗"
        exit 1
    fi
else
    print_warning "跳過步驟 2，使用現有的 Retriever checkpoint"
    RETRIEVER_CKPT=$(get_latest_retriever_checkpoint)
    if [ $? -ne 0 ]; then
        print_error "未找到 Retriever checkpoint"
        exit 1
    fi
fi

# ==================== 步驟 3: ReRanker Training ====================
if [ "$SKIP_STEP3" = false ]; then
    print_info "========== 步驟 3: ReRanker Training =========="
    
    # 3.1 獲取 ReRanker 訓練數據
    if [ "$SKIP_STEP3_1" = false ]; then
        print_info "3.1 獲取 ReRanker 訓練數據..."
        cd "$PROJECT_ROOT"
        
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ranking.py \
            --rank_stage retrieval \
            --data_split $DATA_SPLIT_TRAIN \
            --ret_type dense_tune \
            --retriever_checkpoint "$RETRIEVER_CKPT" \
            --user_emb_path "$USER_EMB_FILE" \
            --CUDA_VISIBLE_DEVICES $CUDA_DEVICE
        
        # 獲取生成的文件名
        RETRIEVAL_FILE_NAME=$(get_latest_retrieval_file_name)
        if [ $? -ne 0 ]; then
            print_error "未找到檢索結果文件"
            exit 1
        fi
        print_success "檢索完成，文件名: $RETRIEVAL_FILE_NAME"
    else
        print_warning "跳過步驟 3.1，使用現有的檢索結果"
        cd "$PROJECT_ROOT"
        RETRIEVAL_FILE_NAME=$(get_latest_retrieval_file_name)
        if [ $? -ne 0 ]; then
            print_error "未找到檢索結果文件"
            exit 1
        fi
        print_success "使用現有檢索結果，文件名: $RETRIEVAL_FILE_NAME"
    fi
    
    # 3.2 生成訓練點
    print_info "3.2 生成訓練點..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generation/generate_point.py \
        --source retrieval \
        --file_name "$RETRIEVAL_FILE_NAME" \
        --CUDA_VISIBLE_DEVICES $CUDA_DEVICE
    
    # 3.3 ReRanker 訓練
    print_info "3.3 ReRanker 訓練..."
    cd rank_tune/reranker
    # 構建完整的用戶嵌入路徑（相對於 rank_tune/reranker 目錄）
    USER_EMB_FULL_PATH="../../data/${TASK}/${DATA_SPLIT_DEV}/${SOURCE}/user_emb/${USER_EMB_FILE}"
    print_info "使用用戶嵌入: $USER_EMB_FULL_PATH"
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python run.py \
        --user_emb_path "$USER_EMB_FULL_PATH"
    cd "$PROJECT_ROOT"
    
    # 驗證 checkpoint
    RERANKER_CKPT=$(get_latest_reranker_checkpoint)
    if [ $? -eq 0 ]; then
        print_success "ReRanker 訓練完成: $RERANKER_CKPT"
    else
        print_error "ReRanker 訓練失敗"
        exit 1
    fi
else
    print_warning "跳過步驟 3，使用現有的 ReRanker checkpoint"
    RERANKER_CKPT=$(get_latest_reranker_checkpoint)
    if [ $? -ne 0 ]; then
        print_error "未找到 ReRanker checkpoint"
        exit 1
    fi
fi

# ==================== 步驟 4: Get Test Result ====================
if [ "$SKIP_STEP4" = false ]; then
    print_info "========== 步驟 4: Get Test Result =========="
    
    cd "$PROJECT_ROOT"
    
    # 4.1 檢索階段
    print_info "4.1 檢索階段..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ranking.py \
        --rank_stage retrieval \
        --data_split $DATA_SPLIT_DEV \
        --ret_type dense_tune \
        --retriever_checkpoint "$RETRIEVER_CKPT" \
        --user_emb_path "$USER_EMB_FILE" \
        --CUDA_VISIBLE_DEVICES $CUDA_DEVICE
    
    # 獲取步驟 4.1 生成的檢索結果文件名和 input_source
    print_info "查找步驟 4.1 生成的檢索結果文件..."
    DEV_RETRIEVAL_FILE_NAME=$(get_latest_retrieval_file_name "$DATA_SPLIT_DEV")
    if [ $? -ne 0 ]; then
        print_error "未找到步驟 4.1 的檢索結果文件"
        exit 1
    fi
    print_success "找到檢索結果文件: $DEV_RETRIEVAL_FILE_NAME"
    
    # 從檢索結果文件路徑中提取 input_source（對應 RetrievalRunner 的 sub_dir）
    # 路徑格式: {output_dir}/{data_split}/{source}/{sub_dir}/retrieval/{file_name}.json
    # 例如: Qwen2-7B-Instruct_outputs/LaMP_2_time/dev/recency/bge-base-en-v1.5_5/retrieval/base_user-6_20251212-113445.json
    DEV_RETRIEVAL_FILE_FULL_PATH=$(find "${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}" -name "${DEV_RETRIEVAL_FILE_NAME}.json" -type f -path "*/retrieval/*" 2>/dev/null | head -1)
    if [ -n "$DEV_RETRIEVAL_FILE_FULL_PATH" ] && [ -f "$DEV_RETRIEVAL_FILE_FULL_PATH" ]; then
        # 提取 input_source：從路徑中提取 retrieval 目錄的父目錄名稱
        RETRIEVAL_DIR=$(dirname "$DEV_RETRIEVAL_FILE_FULL_PATH")  # .../retrieval
        INPUT_SOURCE_FOR_RERANK=$(basename $(dirname "$RETRIEVAL_DIR"))  # bge-base-en-v1.5_5
        print_info "從檢索結果路徑提取 input_source: $INPUT_SOURCE_FOR_RERANK"
    else
        # 如果找不到，使用默認值
        INPUT_SOURCE_FOR_RERANK="bge-base-en-v1.5_5"
        print_warning "無法從路徑提取 input_source，使用默認值: $INPUT_SOURCE_FOR_RERANK"
    fi
    
    # 4.2 重排序階段
    print_info "4.2 重排序階段..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python ranking.py \
        --rank_stage rerank \
        --data_split $DATA_SPLIT_DEV \
        --rerank_type cross_tune \
        --reranker_checkpoint "$RERANKER_CKPT" \
        --input_file "$DEV_RETRIEVAL_FILE_NAME" \
        --input_source "$INPUT_SOURCE_FOR_RERANK" \
        --user_emb_path "$USER_EMB_FILE" \
        --CUDA_VISIBLE_DEVICES $CUDA_DEVICE
    
    # 獲取 rerank 結果文件完整路徑
    # 優先查找包含 rerank 的目錄，按時間排序
    RERANK_FILE_FULL_PATH=$(find "${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}" -name "*.json" -type f -path "*rerank*" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -z "$RERANK_FILE_FULL_PATH" ] || [ ! -f "$RERANK_FILE_FULL_PATH" ]; then
        # 如果沒找到，嘗試查找所有 json 文件
        RERANK_FILE_FULL_PATH=$(find "${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}" -name "*.json" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    fi
    if [ -z "$RERANK_FILE_FULL_PATH" ] || [ ! -f "$RERANK_FILE_FULL_PATH" ]; then
        print_error "未找到 Rerank 結果文件"
        print_error "搜索目錄: ${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}"
        exit 1
    fi
    
    RERANK_FILE_NAME=$(basename "$RERANK_FILE_FULL_PATH" .json)
    print_success "找到 Rerank 結果文件: $RERANK_FILE_NAME"
    print_info "完整路徑: $RERANK_FILE_FULL_PATH"
    
    # 4.3 生成最終結果
    print_info "4.3 生成最終結果..."
    # 從完整路徑中提取 input_source 和 result_name (source)
    # 實際路徑格式: {output_dir}/{data_split}/{source}/{input_source}/{model_name}/{timestamp}_rerank_{topk}/{file_name}.json
    # 例如: .../dev/recency/bge-base-en-v1.5_5/bge-reranker-base/20251212-125020_rerank_5/{file_name}.json
    # 其中 result_name = bge-reranker-base/20251212-125020_rerank_5 (包含模型名稱和時間戳)
    
    RERANK_DIR=$(dirname "$RERANK_FILE_FULL_PATH")  # .../bge-reranker-base/20251212-125020_rerank_5
    RESULT_NAME_WITH_MODEL=$(basename "$RERANK_DIR")  # 20251212-125020_rerank_5
    RESULT_NAME_DIR=$(dirname "$RERANK_DIR")  # .../bge-base-en-v1.5_5/bge-reranker-base
    MODEL_NAME_PART=$(basename "$RESULT_NAME_DIR")  # bge-reranker-base
    RESULT_NAME="${MODEL_NAME_PART}/${RESULT_NAME_WITH_MODEL}"  # bge-reranker-base/20251212-125020_rerank_5
    
    INPUT_SOURCE_DIR=$(dirname "$RESULT_NAME_DIR")  # .../recency/bge-base-en-v1.5_5
    INPUT_SOURCE=$(basename "$INPUT_SOURCE_DIR")  # bge-base-en-v1.5_5
    
    print_info "Rerank 路徑信息:"
    print_info "  完整路徑: $RERANK_FILE_FULL_PATH"
    print_info "  Input Source: $INPUT_SOURCE"
    print_info "  Result Name (Source): $RESULT_NAME"
    print_info "  File Name: $RERANK_FILE_NAME"
    
    # 執行生成並檢查錯誤
    print_info "執行生成..."
    if ! CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python generation/generate.py \
        --task "$TASK" \
        --input_path "${DATA_SPLIT_DEV}/${SOURCE}/${INPUT_SOURCE}/" \
        --source "$RESULT_NAME" \
        --file_name "$RERANK_FILE_NAME" \
        --model_name "$MODEL_NAME" \
        --CUDA_VISIBLE_DEVICES $CUDA_DEVICE; then
        print_error "生成結果時發生錯誤"
        exit 1
    fi
    
    # 查找並顯示最新的評估結果
    print_info "查找評估結果..."
    SCORES_FILE=$(find "${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}" -name "scores_*.json" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    if [ -n "$SCORES_FILE" ] && [ -f "$SCORES_FILE" ]; then
        print_success "========== 評估結果 =========="
        cat "$SCORES_FILE"
        echo ""  # 空行
        print_info "評估結果已保存到: $SCORES_FILE"
        
        # 也顯示預測結果文件位置
        PREDICTIONS_FILE=$(find "${OUTPUT_DIR}/${DATA_SPLIT_DEV}/${SOURCE}" -name "predictions_*.json" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
        if [ -n "$PREDICTIONS_FILE" ] && [ -f "$PREDICTIONS_FILE" ]; then
            print_info "預測結果已保存到: $PREDICTIONS_FILE"
        fi
    else
        print_warning "未找到評估結果文件，可能生成過程未完成"
        print_info "請檢查生成日誌以確認是否有錯誤"
    fi
    
    print_success "測試結果生成完成"
else
    print_warning "跳過步驟 4"
fi

# ==================== 完成 ====================
print_success "========== 所有步驟完成！=========="
print_info "用戶嵌入: $USER_EMB_FILE"
print_info "Retriever Checkpoint: $RETRIEVER_CKPT"
print_info "ReRanker Checkpoint: $RERANKER_CKPT"
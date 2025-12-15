import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

import utils
from arguments import DataArguments, ModelArguments, TrainingArguments


class UserEncoder(nn.Module):

    def __init__(self, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.device = train_args.device


        if data_args.freeze_emb:
            self.corpus_embedding = nn.Embedding.from_pretrained(torch.load(
                data_args.corpus_emb_path),
                                                                 padding_idx=0,
                                                                 freeze=True)
        else:
            self.corpus_embedding = nn.Embedding.from_pretrained(torch.load(
                data_args.corpus_emb_path),
                                                                 padding_idx=0,
                                                                 freeze=False)
        logging.info("load corpus embedding from: {} freeze: {}".format(
            data_args.corpus_emb_path, data_args.freeze_emb))

        self.corpus_trans = nn.Linear(self.corpus_embedding.weight.shape[1],
                                      model_args.emb_dim)

        self.pos = PositionalEmbedding(data_args.max_profile_len,
                                       model_args.emb_dim)
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=model_args.emb_dim,
            nhead=model_args.num_heads,
            dim_feedforward=model_args.emb_dim,
            dropout=model_args.dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=model_args.num_layers)

        # 根據配置決定是否使用注意力池化
        if model_args.use_attention_pooling:
            # 注意力加權池化：學習不同歷史項目的重要性
            self.attention_pooling = AttentionPooling(
                d_model=model_args.emb_dim,
                dropout=model_args.dropout
            )
            self.use_attention = True
        else:
            self.use_attention = False

        # 相似用戶檢索與聚合
        if model_args.use_similar_user_aggregation:
            self.similar_user_retriever = SimilarUserRetriever(
                user_emb_lookup_path=data_args.user_emb_lookup_path,
                device=self.device,
                topk=model_args.similar_user_topk
            )
            
            if model_args.aggregation_method == 'attention':
                self.similar_user_aggregator = SimilarUserAttentionAggregation(
                    d_model=model_args.emb_dim,
                    dropout=model_args.dropout,
                    use_similarity_as_bias=model_args.use_similarity_as_bias
                )
            elif model_args.aggregation_method == 'weighted_sum':
                # 簡單的加權求和（不需要額外的模組，直接在 forward 中實現）
                self.similar_user_aggregator = None
            else:
                raise ValueError("Unknown aggregation_method: {}".format(
                    model_args.aggregation_method))
            
            self.use_similar_user = True
            self.aggregation_method = model_args.aggregation_method
            self.similar_user_weight = model_args.similar_user_weight
            self.enable_dynamic_retrieval = data_args.enable_dynamic_retrieval
            logging.info("Similar user aggregation enabled: topk={}, method={}, weight={}".format(
                model_args.similar_user_topk, model_args.aggregation_method, 
                model_args.similar_user_weight))
        else:
            self.use_similar_user = False
            self.similar_user_retriever = None
            self.similar_user_aggregator = None

        self.infoNCE = InfoNCE(
            batch_size=train_args.per_device_train_batch_size,
            hidden_dim=model_args.emb_dim,
            sim_metric=model_args.sim_metric,
            sim_map=model_args.sim_map,
            sim_activate=model_args.sim_activate,
            infoNCE_temp=model_args.infoNCE_temp,
            infoNCE_temp_learn=model_args.infoNCE_temp_learn,
            device=self.device)
        self.to(self.device)

    def encode_corpus(self, corpus):

        corpus_emb = self.corpus_embedding(corpus)

        if self.corpus_embedding.weight.shape[1] != self.model_args.emb_dim:
            corpus_emb = self.corpus_trans(corpus_emb)

        return corpus_emb

    def forward(self, corpus, corpus_mask):
        corpus = corpus.to(self.device)
        corpus_mask = corpus_mask.to(self.device)

        corpus_emb = self.encode_corpus(corpus).reshape(
            (corpus_mask.shape[0], self.data_args.max_profile_len, -1))

        corpus_emb += self.pos(corpus_emb)
        corpus_encoded = self.transformer_encoder(
            src=corpus_emb, src_key_padding_mask=corpus_mask)

        corpus_encoded = corpus_encoded.masked_fill(corpus_mask.unsqueeze(2),
                                                    0)
        # 根據配置選擇池化方法
        if self.use_attention:
            # 使用注意力加權池化
            corpus_emb_mean = self.attention_pooling(corpus_encoded, corpus_mask)
        else:
            # 使用平均池化（原始方法）
            # 計算有效長度（排除 padding）
            valid_lengths = (~corpus_mask).sum(dim=1, keepdim=True).float()  # [B, 1]
            valid_lengths = valid_lengths.clamp(min=1.0)  # 避免除以 0
            corpus_emb_mean = corpus_encoded.sum(dim=1) / valid_lengths  # [B, d_model]

        # 如果啟用相似用戶聚合（僅在訓練時使用）
        if self.use_similar_user and self.training and self.similar_user_retriever is not None:
            # 檢索相似用戶
            similar_embs, similarity_scores = self.similar_user_retriever.retrieve_topk(
                corpus_emb_mean
            )
            
            if similar_embs is not None:
                # 使用聚合方法
                if self.aggregation_method == 'attention' and self.similar_user_aggregator is not None:
                    # 注意力聚合
                    aggregated_emb = self.similar_user_aggregator(
                        corpus_emb_mean, similar_embs, similarity_scores
                    )
                elif self.aggregation_method == 'weighted_sum':
                    # 簡單加權求和（基於相似度分數）
                    weights = F.softmax(similarity_scores, dim=-1).unsqueeze(-1)  # [B, topk, 1]
                    aggregated_emb = (weights * similar_embs).sum(dim=1)  # [B, D]
                    # 殘差連接
                    aggregated_emb = aggregated_emb + corpus_emb_mean
                else:
                    aggregated_emb = corpus_emb_mean
                
                # 融合原始 embedding 和聚合後的 embedding
                final_emb = (1 - self.similar_user_weight) * corpus_emb_mean + \
                           self.similar_user_weight * aggregated_emb
                return final_emb

        return corpus_emb_mean
    
    def update_similar_user_embeddings(self, new_embeddings):
        """
        更新相似用戶檢索器中的 embeddings（用於動態更新）
        Args:
            new_embeddings: [num_users, emb_dim] - 新的用戶 embeddings
        """
        if self.use_similar_user and self.similar_user_retriever is not None:
            self.similar_user_retriever.update_embeddings(new_embeddings)

    def loss(self, corpus_1, corpus_1_mask, corpus_2, corpus_2_mask):
        corpus_1_emb = self.forward(corpus_1, corpus_1_mask)
        corpus_2_emb = self.forward(corpus_2, corpus_2_mask)
        loss = self.infoNCE(corpus_1_emb, corpus_2_emb)
        return {"total_loss": loss}

    def get_user_emb(self, corpus, corpus_mask):
        return self.forward(corpus, corpus_mask)

    def save_model(self):
        model_path = self.train_args.model_path
        utils.check_dir(model_path)
        logging.info("save model to: {}".format(model_path))
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path=None):
        logging.info("load model from: {}".format(model_path))
        self.load_state_dict(torch.load(model_path, map_location=self.device))


class InfoNCE(nn.Module):

    def __init__(self, batch_size, hidden_dim, sim_metric, sim_map,
                 sim_activate, infoNCE_temp, infoNCE_temp_learn,
                 device) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.sim_metric = sim_metric
        self.sim_map = sim_map
        self.sim_activate = sim_activate

        if infoNCE_temp_learn:
            self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        else:
            self.infoNCE_temp = infoNCE_temp

        if sim_map:
            self.weight_matrix = nn.Parameter(
                torch.randn((hidden_dim, hidden_dim)))
            nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, his_1_mean: torch.Tensor, his_2_mean: torch.Tensor):
        batch_size = his_1_mean.size(0)
        N = 2 * batch_size

        if self.sim_metric == 'cosine':
            his_1_mean = F.normalize(his_1_mean, p=2, dim=-1)
            his_2_mean = F.normalize(his_2_mean, p=2, dim=-1)

        z = torch.cat([his_1_mean, his_2_mean], dim=0)

        if self.sim_map:
            sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        else:
            sim = torch.mm(z, z.T)

        # logging.info(f"type: {type(self.sim_activate)}")
        if self.sim_activate is None:
            sim = sim / self.infoNCE_temp
        elif self.sim_activate == 'tanh':
            sim = torch.tanh(sim) / self.infoNCE_temp
        else:
            logging.info("sim_activate: {}".format(self.sim_activate))
            raise NotImplementedError

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss


class SimilarUserRetriever(nn.Module):
    """
    檢索 top-k 相似用戶的 embeddings
    用於在訓練過程中動態檢索相似用戶並聚合其信息
    """
    def __init__(self, user_emb_lookup_path, device, topk=5):
        super().__init__()
        self.device = device
        self.topk = topk
        self.user_embeddings = None
        
        # 載入所有用戶的 embeddings（用於檢索）
        if user_emb_lookup_path and os.path.exists(user_emb_lookup_path):
            try:
                self.user_embeddings = torch.load(user_emb_lookup_path, map_location=device)
                # 確保 embeddings 是 2D tensor: [num_users, emb_dim]
                if self.user_embeddings.dim() == 1:
                    self.user_embeddings = self.user_embeddings.unsqueeze(0)
                # 正規化 embeddings
                self.user_embeddings = F.normalize(self.user_embeddings, p=2, dim=-1)
                logging.info("Loaded user embeddings for retrieval: shape {}".format(
                    self.user_embeddings.shape))
            except Exception as e:
                logging.warning("Failed to load user embeddings from {}: {}".format(
                    user_emb_lookup_path, e))
                self.user_embeddings = None
        else:
            if user_emb_lookup_path:
                logging.warning("User embedding lookup path does not exist: {}".format(
                    user_emb_lookup_path))
            self.user_embeddings = None
    
    def update_embeddings(self, new_embeddings):
        """
        更新用戶 embeddings（用於動態更新）
        Args:
            new_embeddings: [num_users, emb_dim] - 新的用戶 embeddings
        """
        if new_embeddings is not None:
            self.user_embeddings = F.normalize(new_embeddings, p=2, dim=-1).to(self.device)
    
    @torch.no_grad()
    def retrieve_topk(self, query_embeddings):
        """
        檢索 top-k 相似用戶的 embeddings
        Args:
            query_embeddings: [batch_size, emb_dim] - 當前批次用戶的 embeddings
        Returns:
            similar_embeddings: [batch_size, topk, emb_dim] - top-k 相似用戶的 embeddings
            similarity_scores: [batch_size, topk] - 相似度分數
        """
        if self.user_embeddings is None:
            return None, None
        
        # 正規化查詢 embeddings
        query_normalized = F.normalize(query_embeddings, p=2, dim=-1)
        
        # 計算相似度: query_embeddings [B, D] @ user_embeddings.T [D, N] => [B, N]
        similarities = torch.matmul(query_normalized, self.user_embeddings.T)
        
        # 獲取 top-k（排除自己，所以取 topk+1 然後過濾）
        topk_values, topk_indices = torch.topk(similarities, min(self.topk + 1, similarities.size(-1)), dim=-1)
        
        # 過濾掉自己（相似度為 1.0 的項）
        batch_size = query_embeddings.size(0)
        similar_embeddings_list = []
        similarity_scores_list = []
        
        for i in range(batch_size):
            # 過濾掉相似度過高的項（可能是自己）
            mask = topk_values[i] < 0.9999  # 避免選擇自己
            valid_indices = topk_indices[i][mask][:self.topk]
            valid_scores = topk_values[i][mask][:self.topk]
            
            if len(valid_indices) > 0:
                similar_embs = self.user_embeddings[valid_indices]  # [valid_k, D]
                # 如果不足 topk，用最後一個填充
                if len(valid_indices) < self.topk:
                    padding = similar_embs[-1:].repeat(self.topk - len(valid_indices), 1)
                    similar_embs = torch.cat([similar_embs, padding], dim=0)
                    padding_scores = valid_scores[-1:].repeat(self.topk - len(valid_scores))
                    valid_scores = torch.cat([valid_scores, padding_scores], dim=0)
            else:
                # 如果沒有有效項，使用最相似的 topk 個（可能包含自己）
                similar_embs = self.user_embeddings[topk_indices[i][:self.topk]]
                valid_scores = topk_values[i][:self.topk]
            
            similar_embeddings_list.append(similar_embs)
            similarity_scores_list.append(valid_scores)
        
        similar_embeddings = torch.stack(similar_embeddings_list, dim=0)  # [B, topk, D]
        similarity_scores = torch.stack(similarity_scores_list, dim=0)  # [B, topk]
        
        return similar_embeddings, similarity_scores


class SimilarUserAttentionAggregation(nn.Module):
    """
    使用注意力機制聚合相似用戶的 embeddings
    """
    def __init__(self, d_model, dropout=0.1, use_similarity_as_bias=True):
        super().__init__()
        self.d_model = d_model
        self.use_similarity_as_bias = use_similarity_as_bias
        
        # Query: 當前用戶的 embedding
        # Key & Value: 相似用戶的 embeddings
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.temperature = nn.Parameter(torch.ones([]) * (d_model ** -0.5))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, current_user_emb, similar_user_embs, similarity_scores=None):
        """
        Args:
            current_user_emb: [batch_size, emb_dim] - 當前用戶的 embedding
            similar_user_embs: [batch_size, topk, emb_dim] - 相似用戶的 embeddings
            similarity_scores: [batch_size, topk] - 相似度分數（可選，作為 attention bias）
        Returns:
            aggregated_emb: [batch_size, emb_dim] - 聚合後的 embedding
        """
        batch_size, topk, d_model = similar_user_embs.shape
        
        # 投影
        query = self.query_proj(current_user_emb).unsqueeze(1)  # [B, 1, D]
        key = self.key_proj(similar_user_embs)  # [B, topk, D]
        value = self.value_proj(similar_user_embs)  # [B, topk, D]
        
        # 計算注意力分數
        scores = torch.bmm(query, key.transpose(1, 2)) / self.temperature  # [B, 1, topk]
        
        # 如果提供相似度分數，可以作為 bias 加入
        if self.use_similarity_as_bias and similarity_scores is not None:
            scores = scores + similarity_scores.unsqueeze(1)  # [B, 1, topk]
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, topk]
        attention_weights = self.dropout(attention_weights)
        
        # 加權聚合
        aggregated = torch.bmm(attention_weights, value).squeeze(1)  # [B, D]
        
        # 與原始 embedding 融合（殘差連接）
        output = self.layer_norm(aggregated + current_user_emb)
        
        return output


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class AttentionPooling(nn.Module):
    """
    注意力加權池化：學習不同歷史項目的重要性
    使用可學習的 query 向量來計算每個歷史項目的注意力權重
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 可學習的 query 向量，用於計算注意力分數
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_normal_(self.query)
        
        # 可選：對序列進行線性變換以增強表達能力
        self.value_proj = nn.Linear(d_model, d_model)
        
        # 溫度參數，用於縮放注意力分數（可學習或固定）
        self.temperature = nn.Parameter(torch.ones([]) * (d_model ** -0.5))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, seq_len, d_model] - Transformer 編碼後的序列
            mask: [batch_size, seq_len] - True 表示 padding，需要被 mask
        Returns:
            pooled: [batch_size, d_model] - 注意力加權後的用戶嵌入
        """
        batch_size, seq_len, d_model = x.shape
        
        # 擴展 query 到 batch size
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, d_model]
        
        # 對 value 進行投影（可選，增強表達能力）
        value = self.value_proj(x)  # [B, seq_len, d_model]
        
        # 計算注意力分數：query 與序列的相似度（scaled dot-product attention）
        scores = torch.bmm(query, value.transpose(1, 2))  # [B, 1, seq_len]
        scores = scores / self.temperature
        
        # 將 padding 位置的分數設為負無窮，確保 softmax 後權重為 0
        mask_expanded = mask.unsqueeze(1)  # [B, 1, seq_len]
        scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Softmax 得到注意力權重
        attention_weights = F.softmax(scores, dim=-1)  # [B, 1, seq_len]
        attention_weights = self.dropout(attention_weights)
        
        # 加權求和得到最終的用戶嵌入
        pooled = torch.bmm(attention_weights, value)  # [B, 1, d_model]
        pooled = pooled.squeeze(1)  # [B, d_model]
        
        # Layer normalization 有助於穩定訓練
        pooled = self.layer_norm(pooled)
        
        return pooled


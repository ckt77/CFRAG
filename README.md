<img width="394" height="157" alt="image" src="https://github.com/user-attachments/assets/0b0e5a0a-3103-4ee3-950d-7945d2b71dcc" />2025-Fall 資訊檢索與擷取Final Project.
原ReadMe檔案請參考原作者: https://github.com/TengShi-RUC/CFRAG
我們依照原作者的環境設定來進行本次project, 亦可參考"requirements.txt"建置環境.
基礎實驗設定:
- 資料集:LaMP_2, LaMP_4 (須自行下載至對應資料夾), 我們使用train和dev兩個部分
- LLM: Qwen2-7B-Instruct (為開源模型)
- top-k: 3
- w_t: 0.7

本專案以 LaMP 任務（特別是 LaMP_4：個人化標題生成）為主要場景，探討如何在檢索增強生成（Retrieval-Augmented Generation, RAG）架構下，引入用戶嵌入（User Embedding）與協同過濾訊號，提升對特定使用者的個人化生成效果。我們同時比較多種檢索策略（Random、Recency、BM25、Dense、Dense_tune），並分析其在個人化生成任務上的優缺點。
除了以CFRAG作為baseline外，本專案結合兩個核心要點:
1. 用戶嵌入學習：利用使用者歷史互動／個人簡介，透過 Transformer Encoder 建立序列化的用戶表示，並在原始平均池化基礎上加入注意力池化（Attention Pooling），讓模型自動學習哪些歷史紀錄更重要。
2. 相似用戶聚合（Similar User Aggregation）：先從用戶嵌入空間檢索 top-k 相似用戶，再利用注意力機制將這些相似用戶資訊聚合回當前用戶的表示，以捕捉「群體偏好」與冷啟動情境下的輔助訊號。

<img width="394" height="157" alt="image" src="https://github.com/user-attachments/assets/96aa03fe-1b46-4de8-926b-91507fd17345" />


實驗中發現，單純依賴相似用戶聚合有時會引入噪音，導致 CFRAG 在部分情境下甚至不如 BM25 與 Random baseline，顯示「個人化訊號如何與語義相關性平衡」是關鍵難題。未來可進一步朝動態調整相似用戶數量與融合權重、更精細的相似用戶選擇策略，以及與更強大的 LLM / 多任務訓練結合等方向發展，以提升個人化檢索與生成的穩定性與上限。

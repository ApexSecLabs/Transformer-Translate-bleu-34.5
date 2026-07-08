import torch

class Config:
    # ========== 数据 ==========
    data_path = "data.txt"          # 训练数据路径，格式：英\t中
    max_len = 60                    # 句子最大 token 数（不含 <eos>）
    min_freq = 1                    # 最小词频，保留所有词汇

    # ========== 模型结构 ==========
    d_model = 256                   # 词向量维度
    nhead = 4                       # 多头注意力头数
    num_encoder_layers = 3          # 编码器层数
    num_decoder_layers = 3          # 解码器层数
    dim_feedforward = 1024          # 前馈网络隐藏层维度
    dropout = 0.3                   # Dropout 比例
    max_pos_len = 200               # 位置编码最大长度

    # ========== 训练 ==========
    batch_size = 32
    epochs = 200
    lr = 0.001                      # 初始学习率，会由 Noam 调度器动态调整
    warmup_steps = 4000
    weight_decay = 1e-4             # 权重衰减
    clip_grad = 1.0
    early_stopping_patience = 10    # 验证损失不降时停止的 epoch 数

    # ========== 推理与评估 ==========
    beam_size = 5                   # 束搜索宽度（适当增大可提升质量）
    len_penalty = 0.8               # 长度惩罚
    max_gen_len = 60                # 生成最大长度

    # ========== 设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 占位符（训练时自动填充） ==========
    src_pad_idx = None
    tgt_pad_idx = None
    src_sos_idx = None
    tgt_sos_idx = None
    src_eos_idx = None
    tgt_eos_idx = None
    src_unk_idx = None
    tgt_unk_idx = None

cfg = Config()
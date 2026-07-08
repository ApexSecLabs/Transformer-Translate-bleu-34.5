import re
from collections import Counter
import torch
from torch.utils.data import Dataset
from config import cfg


def tokenize_en(text):
    """英文分词：小写 + 按空格切分"""
    return text.lower().split()


def tokenize_zh_char(text):
    """中文分词：去除所有空白字符后按字切分（字符级模型）"""
    return list(re.sub(r'\s+', '', text))


class TranslationDataset(Dataset):
    """英译中数据集，独立构建源/目标词表，过滤低频词"""
    def __init__(self, data_path, src_tokenizer, tgt_tokenizer, max_len, min_freq):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.pairs = []
        self._load_data(data_path)
        self._build_vocab(min_freq)
        self._convert_to_indices()

    def _load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                src, tgt = parts[0], parts[1]
                src_tokens = self.src_tokenizer(src)[:self.max_len]
                tgt_tokens = self.tgt_tokenizer(tgt)[:self.max_len]
                if src_tokens and tgt_tokens:
                    self.pairs.append((src_tokens, tgt_tokens))

    def _build_vocab(self, min_freq):
        specials = ['<pad>', '<sos>', '<eos>', '<unk>']  # 索引 0,1,2,3
        src_counter = Counter()
        tgt_counter = Counter()
        for src, tgt in self.pairs:
            src_counter.update(src)
            tgt_counter.update(tgt)

        self.src_vocab = {sym: i for i, sym in enumerate(specials)}
        self.tgt_vocab = {sym: i for i, sym in enumerate(specials)}

        for w, f in src_counter.items():
            if f >= min_freq:
                self.src_vocab[w] = len(self.src_vocab)
        for w, f in tgt_counter.items():
            if f >= min_freq:
                self.tgt_vocab[w] = len(self.tgt_vocab)

        # 记录特殊 token 索引
        self.src_unk_idx = self.src_vocab['<unk>']
        self.tgt_unk_idx = self.tgt_vocab['<unk>']
        self.src_pad_idx = self.src_vocab['<pad>']
        self.tgt_pad_idx = self.tgt_vocab['<pad>']
        self.src_sos_idx = self.src_vocab['<sos>']
        self.tgt_sos_idx = self.tgt_vocab['<sos>']
        self.src_eos_idx = self.src_vocab['<eos>']
        self.tgt_eos_idx = self.tgt_vocab['<eos>']

        # 同步到全局配置，供 collate_fn 等使用
        cfg.src_pad_idx = self.src_pad_idx
        cfg.tgt_pad_idx = self.tgt_pad_idx
        cfg.src_sos_idx = self.src_sos_idx
        cfg.tgt_sos_idx = self.tgt_sos_idx
        cfg.src_eos_idx = self.src_eos_idx
        cfg.tgt_eos_idx = self.tgt_eos_idx
        cfg.src_unk_idx = self.src_unk_idx
        cfg.tgt_unk_idx = self.tgt_unk_idx

    def _convert_to_indices(self):
        """将分词后的 token 序列转换为 ID 序列，并添加 <eos>"""
        self.data = []
        for src_tokens, tgt_tokens in self.pairs:
            src_ids = [self.src_vocab.get(w, self.src_unk_idx) for w in src_tokens] + [self.src_eos_idx]
            tgt_ids = [self.tgt_vocab.get(w, self.tgt_unk_idx) for w in tgt_tokens] + [self.tgt_eos_idx]
            self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        # 解码器输入：<sos> + 去掉最后一个 <eos> 的目标序列
        dec_input = [self.tgt_sos_idx] + tgt_ids[:-1]
        dec_output = tgt_ids   # 包含 <eos>
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt_input': torch.tensor(dec_input, dtype=torch.long),
            'tgt_output': torch.tensor(dec_output, dtype=torch.long)
        }


def collate_fn(batch):
    """对 batch 进行 padding，并生成相应的 mask"""
    src_seqs = [item['src'] for item in batch]
    tgt_input_seqs = [item['tgt_input'] for item in batch]
    tgt_output_seqs = [item['tgt_output'] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True,
                                                 padding_value=cfg.src_pad_idx)
    tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input_seqs, batch_first=True,
                                                       padding_value=cfg.tgt_pad_idx)
    tgt_output_padded = torch.nn.utils.rnn.pad_sequence(tgt_output_seqs, batch_first=True,
                                                        padding_value=cfg.tgt_pad_idx)

    src_mask = (src_padded == cfg.src_pad_idx)
    tgt_mask = (tgt_input_padded == cfg.tgt_pad_idx)

    return {
        'src': src_padded.to(cfg.device),
        'tgt_input': tgt_input_padded.to(cfg.device),
        'tgt_output': tgt_output_padded.to(cfg.device),
        'src_mask': src_mask.to(cfg.device),
        'tgt_mask': tgt_mask.to(cfg.device)
    }
import math
import pickle
import argparse
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import cfg
from dataset import tokenize_en
from model import Transformer

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
    print("警告: sacrebleu 未安装，无法计算 BLEU。请运行: pip install sacrebleu")


# ---------------- 解码函数 ----------------
def greedy_decode(model, src, sos_idx, eos_idx, max_len, device):
    """贪心解码（返回去除 <sos> 的 ID 序列）"""
    src_mask = (src == model.src_pad_idx)
    src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.pos_enc(src_emb)
    src_emb = model.dropout(src_emb)
    memory = src_emb
    for layer in model.encoder_layers:
        memory = layer(memory, src_key_padding_mask=src_mask)

    tgt = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_len = tgt.size(1)
        causal_mask = model.generate_square_subsequent_mask(tgt_len, device)
        tgt_emb = model.tgt_embedding(tgt) * math.sqrt(model.d_model)
        tgt_emb = model.pos_enc(tgt_emb)
        tgt_emb = model.dropout(tgt_emb)
        output = tgt_emb
        for layer in model.decoder_layers:
            output = layer(output, memory, tgt_mask=causal_mask,
                           memory_key_padding_mask=src_mask)
        output = model.final_norm(output)
        logits = model.output_proj(output)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == eos_idx:
            break
        tgt = torch.cat([tgt, next_token], dim=1)
    return tgt[0].tolist()[1:]      # 跳过起始 <sos>


def beam_search_decode(model, src, sos_idx, eos_idx, max_len, beam_size, len_penalty, device):
    """束搜索解码（返回去除 <sos> 的 ID 序列）"""
    src_mask = (src == model.src_pad_idx)
    src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.pos_enc(src_emb)
    src_emb = model.dropout(src_emb)
    memory = src_emb
    for layer in model.encoder_layers:
        memory = layer(memory, src_key_padding_mask=src_mask)

    beams = [(0.0, [sos_idx])]
    finished = []
    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == eos_idx:
                finished.append((score, seq))
                continue
            tgt = torch.tensor([seq], device=device)
            tgt_len = tgt.size(1)
            causal_mask = model.generate_square_subsequent_mask(tgt_len, device)
            tgt_emb = model.tgt_embedding(tgt) * math.sqrt(model.d_model)
            tgt_emb = model.pos_enc(tgt_emb)
            tgt_emb = model.dropout(tgt_emb)
            output = tgt_emb
            for layer in model.decoder_layers:
                output = layer(output, memory, tgt_mask=causal_mask,
                               memory_key_padding_mask=src_mask)
            output = model.final_norm(output)
            logits = model.output_proj(output)
            next_logits = logits[0, -1, :]
            log_probs = F.log_softmax(next_logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                new_score = score + topk_log_probs[i].item()
                new_seq = seq + [topk_indices[i].item()]
                new_beams.append((new_score, new_seq))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
    finished.extend(beams)

    # 长度惩罚
    for i, (score, seq) in enumerate(finished):
        finished[i] = (score / (len(seq) ** len_penalty), seq)
    finished.sort(key=lambda x: x[0], reverse=True)
    best_seq = finished[0][1]
    if best_seq[0] == sos_idx:
        best_seq = best_seq[1:]    # 去除起始 <sos>
    return best_seq


# ---------------- 模型加载 ----------------
def load_model_and_vocab(model_path, vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    model = Transformer(
        src_vocab_size=len(vocab['src_vocab']),
        tgt_vocab_size=len(vocab['tgt_vocab']),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.max_pos_len,
        src_pad_idx=vocab['src_pad_idx'],
        tgt_pad_idx=vocab['tgt_pad_idx']
    ).to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    idx2tgt = {v: k for k, v in vocab['tgt_vocab'].items()}
    return model, vocab, idx2tgt


def translate_sentence(model, idx2tgt, vocab, sentence, beam_size=4, len_penalty=0.6, max_len=50):
    """翻译一句英文，返回中文字符串"""
    src_tokens = tokenize_en(sentence)
    src_ids = [vocab['src_vocab'].get(w, vocab['src_unk_idx']) for w in src_tokens] + [vocab['src_eos_idx']]
    src_tensor = torch.tensor([src_ids], device=cfg.device)

    with torch.no_grad():
        if beam_size <= 1:
            tgt_ids = greedy_decode(model, src_tensor, vocab['tgt_sos_idx'], vocab['tgt_eos_idx'],
                                    max_len, cfg.device)
        else:
            tgt_ids = beam_search_decode(model, src_tensor, vocab['tgt_sos_idx'], vocab['tgt_eos_idx'],
                                         max_len, beam_size, len_penalty, cfg.device)

    tgt_text = ''.join([idx2tgt.get(idx, '<unk>') for idx in tgt_ids
                        if idx not in (vocab['tgt_sos_idx'], vocab['tgt_eos_idx'])])
    return tgt_text


# ---------------- BLEU 评估 ----------------
def evaluate_bleu(model, vocab, idx2tgt, data_path, max_samples=1000):
    """从平行数据中随机抽取句子计算 BLEU"""
    if sacrebleu is None:
        print("sacrebleu 未安装，跳过 BLEU 评估。")
        return

    # 读取全部平行句对
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))

    # 随机抽样
    random.seed(42)
    if len(pairs) > max_samples:
        samples = random.sample(pairs, max_samples)
    else:
        samples = pairs

    references = [zh for _, zh in samples]
    hypotheses = []
    for en, _ in tqdm(samples, desc="Evaluating BLEU", leave=False):
        hyp = translate_sentence(model, idx2tgt, vocab, en,
                                 beam_size=cfg.beam_size,
                                 len_penalty=cfg.len_penalty,
                                 max_len=cfg.max_gen_len)
        hypotheses.append(hyp)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='zh')
    print(f"\nBLEU (抽样 {len(samples)} 句): {bleu.score:.2f}")
    print(f"测试数据路径: {data_path}")


# ---------------- 主程序 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_transformer.pt', help='模型文件路径')
    parser.add_argument('--vocab', type=str, default='vocab.pkl', help='词表文件路径')
    parser.add_argument('--test_data', type=str, default=None, help='用于 BLEU 评估的平行数据（默认使用训练数据）')
    parser.add_argument('--max_bleu_samples', type=int, default=1000, help='BLEU 评估抽取的最大句子数')
    parser.add_argument('--no_bleu', action='store_true', help='跳过 BLEU 评估，直接进入交互翻译')
    args = parser.parse_args()

    # 加载模型
    model, vocab, idx2tgt = load_model_and_vocab(args.model, args.vocab)
    print(f"模型加载完成，目标词表大小: {len(vocab['tgt_vocab'])}")

    # BLEU 评估（除非手动禁用）
    if not args.no_bleu:
        test_data = args.test_data if args.test_data else cfg.data_path
        evaluate_bleu(model, vocab, idx2tgt, test_data, max_samples=args.max_bleu_samples)

    # 交互式翻译
    print("\n--- 交互式翻译 (输入 'quit' 退出) ---")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() == 'quit':
            break
        if not sentence:
            continue
        translation = translate_sentence(model, idx2tgt, vocab, sentence,
                                         beam_size=cfg.beam_size,
                                         len_penalty=cfg.len_penalty,
                                         max_len=cfg.max_gen_len)
        print(translation)


if __name__ == "__main__":
    main()
from rouge import Rouge

def evaluate(model, loader, tokenizer, device="cpu"):
    rouge = Rouge()
    model.eval()
    scores = []

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            for i in range(X.size(0)):
                seq_input = X[i][:int(0.75*X.size(1))]
                target_seq = Y[i][int(0.75*Y.size(1)):]
                pred_tokens = model.generate(seq_input, max_len=len(target_seq), device=device)
                pred_text = tokenizer.decode(pred_tokens[-len(target_seq):])
                target_text = tokenizer.decode(target_seq.tolist())
                score = rouge.get_scores(pred_text, target_text)[0]['rouge-l']['f']
                scores.append(score)

    return sum(scores) / len(scores)

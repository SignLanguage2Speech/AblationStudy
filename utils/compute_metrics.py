import numpy as np
import evaluate
import torch
import torch.nn as nn


def compute_metrics(model, dataloaderTest, loss_preds_fc, tokenize_targets, epoch, CFG):

    metrics = {}
    bleu = evaluate.load("bleu")
    metrics[f"BLEU_1"] = 0
    metrics[f"BLEU_2"] = 0
    metrics[f"BLEU_3"] = 0
    metrics[f"BLEU_4"] = 0
    metrics[f"LOSS"] = 0
    rouge = evaluate.load('rouge')
    metrics["ROUGE"] = 0

    preds = []
    targets = []

    example_index = np.random.randint(len(dataloaderTest))
    for j, datapoint in enumerate(dataloaderTest):
        ipt, ipt_len, trg, trg_len, trg_transl, trg_gloss, max_ipt_len = datapoint
        
        tokenized_trg_transl = tokenize_targets(
            trg_transl, 
            model.language_model.tokenizer, 
            "de_DE", 
            model.language_model.max_seq_length, 
            epoch, 
            CFG.epochs,
            CFG.device)

        predicts, loss_ce = model(ipt.to(CFG.device), tokenized_trg_transl, ipt_len)
        preds_permute = predicts.permute(0,2,1)
        
        loss = loss_preds_fc(
                nn.functional.log_softmax(predicts, dim=-1).contiguous().view(-1,predicts.size(-1)), 
                tokenized_trg_transl.contiguous().view(-1)) / ipt.size(0)

        metrics[f"LOSS"] += loss.detach().cpu().numpy()

        raw_preds = model.predict(ipt.to(CFG.device), ipt_len)
        raw_targets = trg_transl

        for i in range(len(raw_preds)):
            targets.append(raw_targets[i])
            if j == example_index:
                metrics[f"EXAMPLE"] = f"pred: {raw_preds[i]}, target: {raw_targets[i]}"
            if raw_preds[i]:
                preds.append(raw_preds[i])
            else:
                preds.append("@")

    metrics[f"BLEU_1"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 1).get("bleu")
    metrics[f"BLEU_2"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 2).get("bleu")
    metrics[f"BLEU_3"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 3).get("bleu")
    metrics[f"BLEU_4"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 4).get("bleu")
    metrics[f"ROUGE"] += rouge.compute(predictions = preds, references = [[target] for target in targets]).get("rouge1")
    metrics[f"LOSS"] /= len(dataloaderTest)
    
    return metrics
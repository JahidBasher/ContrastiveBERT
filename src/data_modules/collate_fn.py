import torch


def collate_fn(batch):

    bn_bert_input, en_bert_input = [], []
    bn_bert_label, en_bert_label = [], []
    bn_segment_label, en_segment_label = [], []

    for items in batch:
        bn_bert_input.append(items[0]['bert_input'])
        en_bert_input.append(items[1]['bert_input'])

        bn_bert_label.append(items[0]['bert_label'])
        en_bert_label.append(items[1]['bert_label'])

        bn_segment_label.append(items[0]['segment_label'])
        en_segment_label.append(items[1]['segment_label'])

    bn_bert_input += en_bert_input
    bn_bert_label += en_bert_label
    bn_segment_label += en_segment_label

    max_length = max(len(seq) for seq in bn_bert_input)

    # Pad sequences in the batch
    padded_bert_inputs = [
        torch.nn.functional.pad(seq, pad=(0, max_length - len(seq)), value=0)
        for seq in bn_bert_input
    ]
    padded_bert_labels = [
        torch.nn.functional.pad(seq, pad=(0, max_length - len(seq)), value=0)
        for seq in bn_bert_label
    ]

    # Stack the padded sequences along the batch dimension
    padded_bert_inputs = torch.stack(padded_bert_inputs)
    padded_bert_labels = torch.stack(padded_bert_labels)
    segment_labels = torch.stack(bn_segment_label)

    # attention_mask = (padded_bert_inputs != 0).float()

    return {
        "bert_input": padded_bert_inputs,
        "bert_label": padded_bert_labels,
        "segment_label": segment_labels,
        # "attention_mask": attention_mask
    }

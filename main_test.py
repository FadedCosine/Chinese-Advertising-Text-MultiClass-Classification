# coding=utf-8

#################################
# file: main_test.py
#################################

import os
import numpy as np
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
from optparse import OptionParser
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
###################################
# predict： 根据模型预测结果并输出结果文件，文件内容格式为qid\t言语\t标签
###################################
def predict(options):

    qid_lst, x_row, y_lst = [], [], []
    read_file = open(options.data_file, 'rb')
    read_file.readline()
    for line in read_file:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split('|')
        x_row.append(line_split[1])
        y_lst.append(int(line_split[0]))
    # for line in read_file:
    #     line = line.decode('utf-8', 'ignore')
    #     line_split = line.strip('\n').strip('\r').split('\t')
    #     qid_lst.append(line_split[0])
    #     x_row.append(line_split[1])
    #     if len(line_split) >=3:
    #         y_lst.append(int(line_split[-1]))
    read_file.close()
    x_test_text, y_test = np.array(x_row), np.array(y_lst)
    print("test text shape is ",x_test_text.shape)
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    MAX_LEN = 180

    test_input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN) for sent in x_test_text]
    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long",
                                   value=0, truncating="post", padding="post")
    test_attention_masks = []
    for sent in test_input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        test_attention_masks.append(att_mask)
    test_inputs = torch.tensor(test_input_ids)
    test_masks = torch.tensor(test_attention_masks)

    batch_size = 2
    # Create the DataLoader for our test set.
    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False,
    )

    # Tell pytorch to run this model on the GPU.
    """加载模型"""

    model.load_state_dict(torch.load(options.model_file))
    model.cuda()
    all_predictions = []
    model.eval()
    for step, batch in enumerate(test_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask 
                    )

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        all_predictions.extend(pred_flat)

    if len(y_test) > 0:
        eps = 1e-3

        TP, FP, FN = 0, 0, 0
        for i in range(len(all_predictions)):
            if all_predictions[i] == y_test[i] and y_test[i] != 0:
                TP += 1
            elif all_predictions[i] == 0 and y_test[i] != 0:
                FN += 1
            elif all_predictions[i] != 0 and y_test[i] == 0:
                FP += 1
        if abs(TP + FP) < eps:
            P = 0.
        else:
            P = float(TP) / float(TP + FP)
        if abs(TP + FP) < eps:
            R = 0.
        else:
            R = float(TP) / float(TP + FN)
        if abs(P) < eps and abs(R) < eps:
            F1 = 0.
        else:
            F1 = 4 * P * R / (P + 3 * R)
        print("TP is ", TP)
        print("FP is ", FP)
        print("FN is ", FN)
        print("Total number of test examples: {}".format(len(y_test)))
        print("P is ", P)
        print("R is ", R)
        print("Score is ",F1)

    print("predict start.......")
    ###################################
    # 预测逻辑和结果输出，("%d\t%s\t%d", qid, content, predict_label)
    ###################################
    print("Saving evaluation to {0}".format(options.out_put_file))
    output_file = open(options.out_put_file, 'wb')
    output_file.write("qid\ttext\tlabel\n".encode("utf-8"))
    for i in range(len(all_predictions)):
        # output_file.write((qid_lst[i]+'\t'+x_row[i]+'\t'+str(int(all_predictions[i]))+'\n').encode('utf-8', 'ignore'))
        output_file.write(
            (x_row[i] + '\t' + str(int(all_predictions[i])) + '\n').encode('utf-8', 'ignore'))
    output_file.close()
    print("predict end.......")

    return None



###################################
# main： 主逻辑
###################################
def main():
    ###################################
    # 读取参数列表
    ###################################
    oparser = OptionParser()

    oparser.add_option("-m", "--model_file", dest="model_file", help="输入模型文件 \
                must be: negative.model", default="./model/AdDec-bert-base-FenciTransAug.pt")

    oparser.add_option("-d", "--data_file", dest="data_file", help="输入验证集文件 \
                must be: validation_data.txt", default="yzx_test")

    oparser.add_option("-o", "--out_put", dest="out_put_file", help="输出结果文件 \
    			must be: result.txt", default="result.txt")

    (options, args) = oparser.parse_args()

    print("main start.....")
    predict(options)
    print("main end.....")
    return 0

if __name__ == '__main__':
    main()

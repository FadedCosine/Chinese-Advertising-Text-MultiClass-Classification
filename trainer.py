# -*- coding: utf-8 -*-
"""BertForAdDec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17JjgKGFGuakbhkbWEH3YulRbAUM3j0tX
"""

import torch
device='cuda' if torch.cuda.is_available() else 'cpu'

import os
import numpy as np

def load_data_and_labels(data_file):
    train_examples = open(data_file, "rb")
    train_examples.readline()
    x = []
    y = []
    max_len = 0
    for line in train_examples:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split(',')
        x.append(''.join(line_split[1].split(' ')))
        max_len = max(max_len, len(x[-1]))
        y.append(int(line_split[0]))
    return np.array(x), np.array(y), max_len
def load_test_data_and_labels(data_file):
    train_examples = open(data_file, "rb")
    train_examples.readline()
    x = []
    y = []
    max_len = 0
    for line in train_examples:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split('|')
        x.append(line_split[1])
        max_len = max(max_len, len(x[-1]))
        y.append(int(line_split[0]))
    return np.array(x), np.array(y), max_len
def load_data_and_labels_with_qid(data_file):
    train_examples = open(data_file, "rb")
    train_examples.readline()
    x = []
    y = []
    max_len = 0
    for line in train_examples:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split('\t')
        # print(line_split)
        x.append(line_split[1])
        if len(x[-1]) >=180:
          print(x[-1])
        max_len = max(max_len, len(x[-1]))
        y.append(int(line_split[-1]))
    return np.array(x), np.array(y), max_len

x_row_text, y_labels, max_len = load_data_and_labels_with_qid("./data/aug_train_data.txt")
print(x_row_text[:10])
print(y_labels[:10])
print(max_len)

from transformers import BertTokenizer, XLNetTokenizer, AutoTokenizer, ElectraTokenizer, AlbertTokenizer, AutoTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
# BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# tokenizer = AlbertTokenizer.from_pretrained('voidful/albert_chinese_large')
# tokenizer = XLNetTokenizer.from_pretrained("hfl/chinese-xlnet-base")
# tokenizer = ElectraTokenizer.from_pretrained("hfl/chinese-electra-base-discriminator")
# Print the original sentence.
print(' Original: ', x_row_text[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(x_row_text[0]))

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x_row_text[0])))
print(max_len)

MAX_LEN=180

input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in x_row_text]

print('Original: ', x_row_text[0])
print('Token IDs:', input_ids[0])

from keras.preprocessing.sequence import pad_sequences
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")


# Create attention masks
attention_masks = []

# For each sentence...
for sent in input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]   
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

"""切分训练集和验证集"""

from sklearn.model_selection import train_test_split

# Use 90% for training and 10% for validation.
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, y_labels, random_state=10, test_size=0.1)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attention_masks, y_labels, random_state=10, test_size=0.1)

"""创建数据集和dataloader"""

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)


train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.

batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

"""创建模型"""

from transformers import BertForSequenceClassification, AdamW, BertConfig, XLNetForSequenceClassification, AutoModelForSequenceClassification, ElectraForTokenClassification, AlbertForSequenceClassification

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 

model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese', 
    num_labels = 5,  
    output_attentions = False, 
    output_hidden_states = False, 
)

# model = XLNetForSequenceClassification.from_pretrained(
#     'hfl/chinese-xlnet-base',
#     num_labels = 5
# )
# model = ElectraForTokenClassification.from_pretrained(
#     'hfl/chinese-electra-base-discriminator',
#     num_labels = 5
# )

# model = AutoModelForSequenceClassification.from_pretrained(
#     'voidful/albert_chinese_large', 
#     num_labels = 5,
#     output_attentions = False, 
#     output_hidden_states = False, 
# )

"""加载模型"""
model.load_state_dict(torch.load('AdDec-bert-base-FenciTransAug.pt'))
# Tell pytorch to run this model on the GPU.
model.cuda()

"""设置optimizer和scheduler"""

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 5

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                      num_warmup_steps = 0, # Default value in run_glue.py
                      num_training_steps = total_steps)

"""写一些函数帮助训练"""

import numpy as np
eps = 1e-3
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
  
def pr_re_f1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    TP, FP, FN = 0, 0, 0
    for i in range(len(pred_flat)):
        if pred_flat[i] == labels_flat[i] and pred_flat[i] != 0:
            TP += 1
        elif pred_flat[i] == 0 and labels_flat[i] != 0:
            FN += 1
        elif pred_flat[i] != 0 and labels_flat[i] == 0:
            FP += 1
    if abs(TP+FP)<eps:
        P = 0.
    else:
        P = float(TP) / float(TP + FP)
    if abs(TP+FN)<eps:
        R = 0.
    else: 
        R = float(TP) / float(TP + FN)
    if abs(P) < eps and abs(R) < eps:
        F1 = 0.
    else:
        F1 = 4 * P * R / (P + 3 * R)
    return P, R, F1

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
# Set the seed value all over the place to make this reproducible.
seed_val = 10

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# Store the average loss after each epoch so we can plot them.
loss_values = []
best_F1 = 0.0
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)
        model.zero_grad()   

        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        loss = outputs[0]
         # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}, loss: {}.'.format(step, len(train_dataloader), elapsed, loss.item()))
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradi ents, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:4f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
        
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        tmp_eval_precision, tmp_eval_recall, tmp_eval_f1 = pr_re_f1(logits, label_ids)
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        eval_precision += tmp_eval_precision
        eval_recall += tmp_eval_recall
        eval_f1 += tmp_eval_f1
        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print("  Precisoin: {0:.4f}".format(eval_precision/nb_eval_steps))
    print("  Recall: {0:.4f}".format(eval_recall/nb_eval_steps))
    print("  F1: {0:.4f}".format(eval_f1/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    if eval_f1/nb_eval_steps > best_F1:
        print("Saving model!")
        best_F1 = eval_f1/nb_eval_steps
        torch.save(model.state_dict(), 'model/AdDec-bert-base-FenciAug.pt')


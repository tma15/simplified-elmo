import csv

import numpy as np
import torch


def convert_data(input_file, vocab, char_vocab, update_vocab):
    processed_data = []
    unk_id = vocab.get_index('<UNK>')
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            words = ['<S>'] + line.strip().split() + ['</S>']

            token_ids = []
            char_ids_list = []
            for word in words:
                if vocab.has_item(word):
                    index = vocab.get_index(word)
                else:
                    index = unk_id
                token_ids.append(index)

                char_ids = char_vocab.item_to_char_ids(word)
                char_ids_list.append(char_ids)

            processed_data.append((token_ids, char_ids_list))
            if len(processed_data) >= 300:
                break
    return processed_data


def convert_ag_news_csv(input_file, vocab, char_vocab, update_vocab):
    processed_data = []
    unk_id = vocab.get_index('<UNK>')
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            label = int(row[0])
            body = row[2]
            words = ['<S>'] + body.strip().split() + ['</S>']

            token_ids = []
            char_ids_list = []
            for word in words:
                if vocab.has_item(word):
                    index = vocab.get_index(word)
                else:
                    index = unk_id
                token_ids.append(index)

                char_ids = char_vocab.item_to_char_ids(word)
                char_ids_list.append(char_ids)

            processed_data.append((token_ids, char_ids_list, label))
            if len(processed_data) >= 300:
                break
    return processed_data


def get_batch(sentences, batch_size, num_steps, max_word_length, device):

    n_batches = int(len(sentences) / batch_size)

    for batch_no in range(n_batches):
        start = batch_size * batch_no
        end = min(batch_size * (batch_no + 1), len(sentences))
        batch_size_t = end - start

        sentences_t = sentences[start: end]

        if num_steps is None:
            max_sentence_length = max([len(s[0]) for s in sentences_t])
        else:
            max_sentence_length = max([len(s[0]) for s in sentences_t])
            max_sentence_length = min(num_steps, max_sentence_length)

        inputs = np.zeros([batch_size_t, max_sentence_length], np.int32)
        char_inputs = np.zeros([batch_size_t, max_sentence_length, max_word_length], np.int32)
        targets = np.zeros([batch_size_t, max_sentence_length], np.int32)

        for i in range(batch_size_t):

            for tok_pos in range(len(sentences_t[i][0]) - 1):
                if tok_pos >= max_sentence_length:
                    break

                inputs[i, tok_pos] = sentences_t[i][0][tok_pos]

                for char_pos in range(len(sentences_t[i][1][tok_pos])):
                    char_inputs[i, tok_pos, char_pos] = sentences_t[i][1][tok_pos][char_pos]

                targets[i, tok_pos] = sentences_t[i][0][tok_pos + 1]

        yield {'token_ids': torch.LongTensor(inputs).to(device),
               'token_characters': torch.LongTensor(char_inputs).to(device),
               'next_token_ids': torch.LongTensor(targets).to(device)}


def get_batch_classification(sentences, batch_size, num_steps, max_word_length):

    n_batches = int(len(sentences) / batch_size)

    for batch_no in range(n_batches):
        start = batch_size * batch_no
        end = min(batch_size * (batch_no + 1), len(sentences))
        batch_size_t = end - start

        sentences_t = sentences[start: end]

        if num_steps is None:
            max_sentence_length = max([len(s[0]) for s in sentences_t])
        else:
            max_sentence_length = max([len(s[0]) for s in sentences_t])
            max_sentence_length = min(num_steps, max_sentence_length)

        inputs = np.zeros([batch_size_t, max_sentence_length], np.int32)
        char_inputs = np.zeros([batch_size_t, max_sentence_length, max_word_length], np.int32)
        targets = np.zeros([batch_size_t], np.int32)

        for i in range(batch_size_t):

            for tok_pos in range(len(sentences_t[i][0]) - 1):
                if tok_pos >= max_sentence_length:
                    break

                inputs[i, tok_pos] = sentences_t[i][0][tok_pos]

                for char_pos in range(len(sentences_t[i][1][tok_pos])):
                    char_inputs[i, tok_pos, char_pos] = sentences_t[i][1][tok_pos][char_pos]

                targets[i] = sentences_t[i][2]

        yield {'token_ids': torch.LongTensor(inputs),
               'token_characters': torch.LongTensor(char_inputs),
               'label_ids': torch.LongTensor(targets)}


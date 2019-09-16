import argparse
import csv
import time

import torch
from torch import optim
import torchtext
from torchtext.utils import download_from_url

import data
from net import Classifier
from vocabulary import UnicodeCharVocabulary
from vocabulary import Vocabulary


# def download():
#     url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms'
#     download_from_url(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', dest='batch_size', default=10, type=int)
    ap.add_argument('--clip', default=10., type=float)
    ap.add_argument('--gpu', default=-1, type=int)
    ap.add_argument('--max_word_length', default=50, type=int)
    ap.add_argument('--max_epoch', default=50, type=int)
    ap.add_argument('--num_steps', default=20, type=int)
    ap.add_argument('--input_file')
    ap.add_argument('--test_file')
    ap.add_argument('--lr', default=0.2, type=float)
    ap.add_argument('--vocab_file')
    args = ap.parse_args()

    hidden_size = 256
    class_size = 5
    cls = Classifier(hidden_size, class_size)

    vocab = Vocabulary()
    vocab.load(args.vocab_file)
    char_vocab = UnicodeCharVocabulary()

    if args.gpu > -1 and torch.cuda.is_available():
        cls = cls.to('cuda')

    optimizer = optim.Adagrad(cls.parameters(), lr=args.lr)

    train_sentences = data.convert_ag_news_csv(args.input_file, vocab, char_vocab, True)
    test_sentences = data.convert_ag_news_csv(args.test_file, vocab, char_vocab, False)

    def test(model, test_data):
        model.eval()
        total = 0
        for batch in data.get_batch_classification(test_data,
                                                   args.batch_size,
                                                   args.num_steps,
                                                   args.max_word_length):
            
            optimizer.zero_grad()
#             loss = model.forward_loss(batch['token_characters'], batch['label_ids'])
            predicted = model(batch['token_characters'])
            result = (batch['label_ids'] == predicted)
            match = result.sum()
            total += len(result)
#             print(batch['label_ids'], predicted)
#             print(a)
#             print()
#             test_loss += loss.item() / batch['token_characters'].size(0)
        accuracy = match / total
        model.train()
        return accuracy

    for e in range(args.max_epoch):
        loss_epoch = 0.
        start_at = time.time()
        for batch in data.get_batch_classification(train_sentences,
                                                   args.batch_size,
                                                   args.num_steps,
                                                   args.max_word_length):

            loss = cls.forward_loss(batch['token_characters'], batch['label_ids'])

            loss_epoch += loss.item() / batch['token_characters'].size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cls.parameters(), args.clip)

            optimizer.step()

            test(cls, test_sentences)
            torch.save(cls.state_dict(), 'classifier.pt')

        duration = time.time() - start_at
#         test_acc = test(cls, test_sentences)
        test_acc = test(cls, traint_sentences)

        print('%d\t%.2f\t%.3f\t%.3f' % (e, duration, loss_epoch, test_acc))

    

if __name__ == '__main__':
    main()

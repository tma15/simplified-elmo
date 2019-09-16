import argparse
import time

import torch
from torch import optim

import data
from elmo.lm import LanguageModel
from vocabulary import Vocabulary
from vocabulary import UnicodeCharVocabulary


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

    vocab = Vocabulary()
    vocab.load(args.vocab_file)
    char_vocab = UnicodeCharVocabulary()

    lm = LanguageModel(vocab)

    if args.gpu > -1 and torch.cuda.is_available():
        lm = lm.to('cuda')

    optimizer = optim.Adagrad(lm.parameters(), lr=args.lr)

    train_sentences = data.convert_data(args.input_file, vocab, char_vocab, True)
    test_sentences = data.convert_data(args.test_file, vocab, char_vocab, False)

    def test(model, test_data):
        model.eval()
        test_loss = 0.
        for batch in data.get_batch(test_data,
                                    args.batch_size,
                                    args.num_steps,
                                    args.max_word_length):
            
            optimizer.zero_grad()
            loss = lm.forward_loss(batch['token_characters'], batch['next_token_ids'])
            test_loss += loss.item() / batch['token_characters'].size(0)
        model.train()
        return test_loss

    for e in range(args.max_epoch):
        loss_epoch = 0.
        start_at = time.time()
        for batch in data.get_batch(train_sentences,
                                    args.batch_size,
                                    args.num_steps,
                                    args.max_word_length):

            optimizer.zero_grad()
            loss = lm.forward_loss(batch['token_characters'], batch['next_token_ids'])
            loss_epoch += loss.item() / batch['token_characters'].size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), args.clip)

            optimizer.step()

            torch.save(lm.elmo.state_dict(), 'elmo-checkpoint.pt')

        duration = time.time() - start_at
        test_loss = test(lm, test_sentences)

        print('%d\t%.2f\t%.3f\t%.3f' % (e, duration, loss_epoch, test_loss))


if __name__ == '__main__':
    main()

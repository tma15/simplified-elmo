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

    device = 'cpu'
    if args.gpu > -1 and torch.cuda.is_available():
        device = 'cuda:%d' % args.gpu
    lm = lm.to(device)

    optimizer = optim.Adagrad(lm.parameters(), lr=args.lr)

    print('#loading training data')
    train_sentences = data.convert_data(args.input_file, vocab, char_vocab, True)
    print('#loading test data')
    test_sentences = data.convert_data(args.test_file, vocab, char_vocab, False)

    def test(model, test_data):
        model.eval()
        test_loss = 0.
        seen_batches = 0
        for batch in data.get_batch(test_data,
                                    args.batch_size,
                                    args.num_steps,
                                    args.max_word_length,
                                    device):
            
            optimizer.zero_grad()
            loss = lm.forward_loss(batch['token_characters'], batch['next_token_ids'])
            seen_batches += 1
            test_loss += loss.item() / batch['token_characters'].size(0)
        model.train()
        return test_loss / seen_batches

    print('#training data', len(train_sentences))
    for e in range(args.max_epoch):
        loss_epoch = 0.
        start_at = time.time()
        seen_batches = 0
        for i, batch in enumerate(data.get_batch(train_sentences,
                                                 args.batch_size,
                                                 args.num_steps,
                                                 args.max_word_length,
                                                 device,
                                                 shuffle=True)):

            optimizer.zero_grad()
            loss = lm.forward_loss(batch['token_characters'], batch['next_token_ids'])
            loss_epoch += loss.item() / batch['token_characters'].size(0)
            seen_batches += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), args.clip)

            optimizer.step()

            if (i + 1) % 1000 == 0:
                duration = time.time() - start_at
                print('epoch %d\titer %d\t%.2f\t%.3f' % (e, i + 1, duration,
                      loss_epoch / seen_batches))

        duration = time.time() - start_at
        test_loss = test(lm, test_sentences)

        print('------------------------')
        print('epoch:%d\ttrain loss:%.3f\ttest loss:%.3f\telapsed::%.2f' % 
              (e, loss_epoch / seen_batches, test_loss, duration))
        print('------------------------')
        torch.save(lm.elmo.state_dict(), 'elmo-checkpoint.pt')


if __name__ == '__main__':
    main()

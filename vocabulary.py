
class Vocabulary:
    def __init__(self):
        self.item_to_index = {}
        self.index_to_item = []

    def __len__(self):
        return len(self.item_to_index)

    def add_item(self, item):
        i = len(self.item_to_index)
        self.index_to_item.append(item)
        self.item_to_index[item] = i

    def get_index(self, item):
        return self.item_to_index[item]

    def get_item(self, index):
        return self.index_to_item[index]

    def has_item(self, item):
        return item in self.item_to_index.keys()

    def load(self, vocab_file):
        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                item = line.strip()
                self.add_item(item)
                if i >= 999:
                    print('vocabulary size is 1000')
                    break


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class UnicodeCharVocabulary:
    def __init__(self):
        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.beginning_of_sentence_character = 256  # <begin sentence>
        self.end_of_sentence_character = 257  # <end sentence>
        self.beginning_of_word_character = 258  # <begin word>
        self.end_of_word_character = 259  # <end word>
        self.padding_character = 260 # <padding>

        self.bos_token = '<S>'
        self.eos_token = '</S>'
        self.max_word_length = 50

        self.beginning_of_sentence_characters = _make_bos_eos(
                self.beginning_of_sentence_character,
                self.padding_character,
                self.beginning_of_word_character,
                self.end_of_word_character,
                self.max_word_length
        )
        self.end_of_sentence_characters = _make_bos_eos(
                self.end_of_sentence_character,
                self.padding_character,
                self.beginning_of_word_character,
                self.end_of_word_character,
                self.max_word_length
        )

    def item_to_char_ids(self, item):
        if item == self.bos_token:
            char_ids = self.beginning_of_sentence_characters
        elif item == self.eos_token:
            char_ids = self.end_of_sentence_characters
        else:
            word_encoded = item.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = self.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]


# coding=utf-8
"""
语料处理类
illool@163.com
QQ:122018919
"""
import corpus
from bert import tokenization
from InputExample import InputExample
from DataProcessor import DataProcessor as Parent


class NerProcessor(Parent):
    def __init__(self):
        self.corpus_ = corpus.get_corpus()

    def get_train_examples(self):
        self._read_data("train_corpus_rootdir_train")
        return self._create_example("train")

    def get_dev_examples(self):
        self._read_data("train_corpus_rootdir_dev")
        return self._create_example("dev")

    def get_test_examples(self):
        self._read_data("train_corpus_rootdir_test")
        return self._create_example("test")

    def get_labels(self):
        return ['M_ORG', 'M_T', 'E_LOC',
                'B_ORG', 'O', 'E_T', 'E_PER',
                'E_ORG', 'W_LOC', 'B_PER',
                'M_PER', 'W_PER', 'M_LOC', 'W_T', 'B_T', 'B_LOC', '[CLS]', '[SEP]']

    def _read_data(self, dir_):
        corpus_ = corpus.get_corpus()
        corpus_.initialize_direct(dir_)
        # return corp.word_seq, corp.tag_seq, corp.all_labels

    @classmethod
    def _read_data_old(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contends) == 0 and words[-1] == '.':
                    l_ = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l_, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines

    def _create_example(self, set_type):
        examples = []
        for (i, line) in enumerate(self.corpus_.word_seq):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(" ".join(self.corpus_.word_seq[i]))
            label = tokenization.convert_to_unicode(" ".join(self.corpus_.tag_seq[i]))
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples

    @classmethod
    def _create_example_old(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples


if __name__ == '__main__':
    corp = corpus.get_corpus()
    # corpus.initialize(bak=False)
    '''
    corp.initialize_direct("train_corpus_rootdir_train", bak=False)
    for i_ in range(len(corp.word_seq)):
        print(corp.word_seq[i_])
        print(corp.tag_seq[i_])
    print(len(corp.word_seq))
    print(corp.all_labels)
    '''
    ner = NerProcessor()
    # ner._read_data("train_corpus_rootdir_test")
    # ner._read_data_old("/data/BERT-NER-master/NERdata/test.txt")
    # ner._create_example("test")
    # ner._create_example_old(line, "test")

    ner.get_train_examples()
    ll = ner.get_labels()
    print(type(ll), ll)
    ner.get_test_examples()
    print(ner.get_labels())
    ner.get_dev_examples()
    print(ner.get_labels())

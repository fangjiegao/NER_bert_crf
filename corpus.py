# coding=utf-8
"""
语料处理的封装
illool@163.com
QQ:122018919
"""
import re
from config import get_config
from util import q_to_b
import os

__corpus = None


# 完全是一个工具类,所以在使用的时候不需要实例化
class Corpus(object):

    _config = get_config()
    _maps = {u't': u'T',
             u'nr': u'PER',
             u'ns': u'ORG',
             u'nt': u'LOC'}

    tag_seq = None  # 样本标签序列
    word_seq = None  # 样本词序列
    all_labels = None  # 标签集

    @classmethod
    def pre_process(cls, dir_config, bak=False):
        """
        语料预处理
        """
        # train_corpus_path = cls._config.get('ner', 'train_corpus_path')
        # lines = cls.read_corpus_from_file(train_corpus_path)

        lines = []
        rootdir = cls._config.get('ner', dir_config)
        file_list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for file in file_list:
            path = os.path.join(rootdir, file)
            if os.path.isfile(path) and path.endswith(".txt"):
                lines += cls.read_corpus_from_file(path)

        new_lines = []
        for line in lines:
            words = q_to_b(line.encode('utf-8').decode('utf-8').strip()).split(u'  ')
            pro_words = cls.process_t(words)
            pro_words = cls.process_nr(pro_words)
            pro_words = cls.process_k(pro_words)
            # print(pro_words[1:])
            new_lines.append('  '.join(pro_words[1:]))
        process_corpus_path = cls._config.get('ner', 'process_corpus_path')
        # 每行文本中加入换行符'/n'
        if bak:
            cls.write_corpus_to_file(data='\n'.join(new_lines).encode('utf-8'), file_path=process_corpus_path)
        return new_lines

    @classmethod
    def process_k(cls, words):
        """
        处理大粒度分词
        """
        pro_words = []
        index = 0
        temp = u''
        while True:
            word = words[index] if index < len(words) else u''
            if u'[' in word and ']' in word:
                word = word.replace(u'[', u'')
                temps = word.split(u']')
                pro_words.append(temps[0])
                temp = u''
            elif u'[' in word:
                temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word.replace(u'[', u''))
            elif u']' in word:
                w = word.split(u']')
                temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=w[0])
                pro_words.append(temp + u'/' + w[1])
                temp = u''
            elif temp:
                temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    @classmethod
    def process_nr(cls, words):
        """
        处理姓名 '/nr'标签合并
        """
        pro_words = []
        index = 0
        while True:
            word = words[index] if index < len(words) else u''
            if u'/nr' in word:
                next_index = index + 1
                if next_index < len(words) and u'/nr' in words[next_index]:
                    pro_words.append(word.replace(u'/nr', u'') + words[next_index])
                    index = next_index
                else:
                    pro_words.append(word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    @classmethod
    def process_t(cls, words):
        """
        处理时间 '/t'标签合并
        """
        pro_words = []
        index = 0
        temp = u''
        while True:
            word = words[index] if index < len(words) else u''
            if u'/t' in word:
                temp = temp.replace(u'/t', u'') + word
            elif temp:
                pro_words.append(temp)
                pro_words.append(word)
                temp = u''
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    @classmethod
    def pos_to_tag(cls, p):
        """
        由词性提取标签
        """
        t = cls._maps.get(p, None)
        return t if t else u'O'

    @classmethod
    def tag_perform_bmeow(cls, tag, index, words):
        """
        标签使用BMEOW模式
        """
        if cls.all_labels is None:
            cls.all_labels = set()
        if index == 0 and tag != u'O':
            if len(words) != 1:
                cls.all_labels.add(u'B_{}'.format(tag))
                return u'B_{}'.format(tag)
            else:
                cls.all_labels.add(u'W_{}'.format(tag))
                return u'W_{}'.format(tag)
        elif 0 < index < len(words)-1 and tag != u'O':
            cls.all_labels.add(u'M_{}'.format(tag))
            return u'M_{}'.format(tag)
        elif index == len(words)-1 and tag != u'O':
            cls.all_labels.add(u'E_{}'.format(tag))
            return u'E_{}'.format(tag)
        else:
            cls.all_labels.add(tag)
            return tag

    @classmethod
    def initialize(cls, bak=False):
        """
        初始化
        """
        corpus_path = cls._config.get('ner', 'process_corpus_path')
        lines = cls.read_corpus_from_file(corpus_path)
        # 数据样本是/table分割的
        words_list = [line.strip().split('  ') for line in lines if line.strip()]
        del lines
        cls.init_sequence(words_list, bak)

    @classmethod
    def initialize_direct(cls, dir_config, bak=False):
        """
        初始化
        """
        if cls.all_labels is not None:
            cls.all_labels.clear()
        if cls.tag_seq is not None:
            cls.tag_seq.clear()
        if cls.word_seq is not None:
            cls.word_seq.clear()
        lines = cls.pre_process(dir_config, bak)
        # 数据样本是/table分割的
        words_list = [line.strip().split('  ') for line in lines if line.strip()]
        del lines
        cls.init_sequence(words_list, bak)

    @classmethod
    def init_sequence(cls, words_list, bak=False):
        """
        初始化字序列、词性序列、标记序列 
        """
        words_seq = [[word.split(u'/')[0] for word in words] for words in words_list]  # [0]:word
        pos_seq = [[word.split(u'/')[1] for word in words] for words in words_list]  # [1]:标签
        tag_seq = [[cls.pos_to_tag(p) for p in pos] for pos in pos_seq]
        cls.tag_seq = [[
            [cls.tag_perform_bmeow(tag_seq[index][i], w, words_seq[index][i])
             for w in range(len(words_seq[index][i]))] for i in range(len(tag_seq[index]))]
             for index in range(len(tag_seq))]
        cls.tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in cls.tag_seq]
        cls.word_seq = [[w for word in word_seq for w in word] for word_seq in words_seq]
        if bak:
            process_corpus_path = cls._config.get('ner', 'processed_corpus_path')
            cls.merge_write_corpus_to_file(file_path=process_corpus_path)
        del words_seq
        del pos_seq
        del tag_seq
        del words_list

    @classmethod
    def read_corpus_from_file(cls, file_path):
        """
        读取语料
        """
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return lines

    @classmethod
    def write_corpus_to_file(cls, data, file_path):
        """
        写语料
        """
        print(file_path)
        f = open(file_path, 'w')
        f.write(data.decode())
        f.close()

    @classmethod
    def merge_write_corpus_to_file(cls, file_path):
        """
        写语料,词:标签
        cls.word_seq
        cls.tag_seq
        """
        print(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'a+') as f:
            for index in range(len(cls.word_seq)):
                f.write(' '.join(cls.word_seq[index]) + '\n')
                f.write(' '.join(cls.tag_seq[index]) + '\n')

    def __init__(self):
        # raise Exception("This class have not element method.")
        pass


def get_corpus():
    """
    单例语料获取
    """
    global __corpus
    if not __corpus:
        # 完全是一个工具类,所以在使用的时候不需要实例化
        __corpus = Corpus
    return __corpus


if __name__ == '__main__':
    corpus = get_corpus()
    # corpus.initialize(bak=False)
    corpus.initialize_direct("train_corpus_rootdir_train", bak=False)
    for i_ in range(len(corpus.word_seq)):
        print(corpus.word_seq[i_])
        print(corpus.tag_seq[i_])
    print(len(corpus.word_seq))
    print(corpus.all_labels)

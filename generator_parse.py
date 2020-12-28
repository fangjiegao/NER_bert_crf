# coding=utf-8

"""
语料生成与解析
illool@163.com
QQ:122018919
"""
import tensorflow as tf
import collections
from InputFeatures import InputFeatures
from PaddingInputExample import PaddingInputExample
import bert.tokenization


# 高效的从.tf_record中解析数据,给模型喂数据
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    # .tf_record中数据的格式
    name_to_features = {  # all features must same with file_based_convert_examples_to_features()
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features_):
        """Decodes a record to a TensorFlow example."""
        # 解析tfrecord文件的每条记录(record),即序列化后的tf.train.Example
        example = tf.parse_single_example(record, name_to_features_)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32. if not use TPU just return example in here
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        # 这里的标签和特征一起封装在 dataset里了,所以model_fn函数中没有了lable参数,该参数也没有使用
        return d

    return input_fn


# 将数据保存为TFRecord file,后续好实现高效的数据吞吐
def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            # tfrecord支持写入三种格式的数据：string，int64，float32，
            # 以列表的形式分别通过tf.train.BytesList、tf.train.Int64List、tf.train.FloatList写入tf.train.Feature
            # tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()]))
            # feature一般是多维数组，要先转为list
            # tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.shape)))
            # tostring函数后feature的形状信息会丢失，把shape也写入
            # tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=[0] * max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)  # 这里主要是将中文分字
    label_a = example.label.split(" ")

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
        label_a = label_a[0:(max_seq_length - 2)]

    tokens_a.insert(0, "[CLS]")
    tokens_a.insert(len(tokens_a), "[SEP]")
    segment_ids = [0] * len(tokens_a)
    label_a.insert(0, "[CLS]")
    label_a.insert(len(tokens_a), "[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)

    input_mask = [1] * len(input_ids)

    label_id = [label_map[_] for _ in label_a]

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_id.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_id) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join(
            [bert.tokenization.printable_text(x) for x in tokens_a]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature

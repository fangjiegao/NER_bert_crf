# coding=utf-8
# Copyright 2019 The BM AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BERT finetuning runner.
illool@163.com
QQ:122018919
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import bert.modeling
import bert.tokenization
import tensorflow as tf
from NerProcessor import NerProcessor
import generator_parse
import model_builder

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", "./",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "/data/bert-checkpoint/chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "ner_pd", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "/data/bert-checkpoint/chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./",
    "The output directory where the model checkpoints will be written.模型的存储路径")

# Other parameters
flags.DEFINE_string(  # None == "/data/bert-checkpoint/chinese_L-12_H-768_A-12/bert_model.ckpt"
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).BERT model的路径")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    " ", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ner_pd": NerProcessor,
    }

    bert.tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                       FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = bert.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:  # max_seq_length < 512
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # estimator的超参数
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,  # 模型路径
        save_checkpoints_steps=FLAGS.save_checkpoints_steps  # 每训练多少步保存
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_builder.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # 训练estimator超参数
    estimator_params = {"batch_size": FLAGS.train_batch_size,
                        "max_seq_length": FLAGS.max_seq_length}
    '''
    tf.estimator.Estimator(
    model_fn, #模型函数
    model_dir=None, #存储目录,训练和验证产生的文件都会存储在这个目录下
    config=None, #设置参数对象,主要针对运行环境的一些设置
    params=None, #超参数，将传递给model_fn使用
    warm_start_from=None #热启动目录路径
    )
    '''
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    # estimator = tf.contrib.tpu.TPUEstimator(
    estimator = tf.estimator.Estimator(
        # use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        params=estimator_params,
        config=run_config
        # train_batch_size=FLAGS.train_batch_size,
        # eval_batch_size=FLAGS.eval_batch_size,
        # predict_batch_size=FLAGS.predict_batch_size
        )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if tf.gfile.Exists(train_file):
            pass
        else:
            generator_parse.file_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
            tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = generator_parse.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples()
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        generator_parse.file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = generator_parse.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results to eval_results.txt*****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                print(key, result[key])
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        label_map_predict = {}
        for (i, label) in enumerate(label_list):
            label_map_predict[i] = label

        predict_examples = processor.get_test_examples()
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        generator_parse.file_based_convert_examples_to_features(predict_examples, label_list,
                                                                FLAGS.max_seq_length, tokenizer,
                                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = generator_parse.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results to test_results.txt*****")
            for (i, prediction) in enumerate(result):
                label_actual = [str(label_map_predict[_]).ljust(6, ' ') for _ in prediction["label_ids"]]
                label_predicts = [str(label_map_predict[_]).ljust(6, ' ') for _ in prediction["predicts"]]
                words_ids = prediction["input_ids"]
                if i >= num_actual_predict_examples:
                    break
                words = tokenizer.convert_ids_to_tokens(words_ids)
                words = [str(_).ljust(6, ' ') for _ in words]
                label_actual_str = "".join(label_actual)
                label_predicts_str = "".join(label_predicts)
                words_str = "".join(words)
                output_line = str(i) + ":" + "\n" + str(len(label_actual)) + ":" + \
                    str(len(label_predicts)) + "\n" + words_str + "\n" + \
                    label_actual_str + "\n" + label_predicts_str + "\n"
                writer.write(output_line)
                num_written_lines += 1

                print(output_line)
                # print(i, prediction)
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()

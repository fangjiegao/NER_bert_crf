# coding=utf-8
"""
模型构建类
illool@163.com
QQ:122018919
"""
import tensorflow as tf
import bert.modeling
import bert.optimization
import numpy as np
import tensorflow.contrib.crf as crf


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, max_seq_length, batch_size):
    """Creates a classification model."""
    model = bert.modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # used = tf.sign(tf.abs(input_ids))
    # lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    output_layer = model.get_sequence_output()  # (64, 128, 768)

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        # shape_list = output_layer.get_shape().as_list()
        print("shape_list:", output_layer.shape, batch_size, max_seq_length)
        sequence_lengths_t = tf.constant(np.full(batch_size, max_seq_length))

        output_layer = tf.reshape(output_layer, [-1, hidden_size])  # [64*128, 768]
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # [64*128, num_labels]
        logits = tf.nn.bias_add(logits, output_bias)  # [64*128, num_labels]
        # logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])  # [64, 128, num_labels]
        logits = tf.reshape(logits, [-1, max_seq_length, num_labels])  # [64, 128, num_labels]

        log_likelihood, transition_params = crf.crf_log_likelihood(logits, labels, sequence_lengths_t)
        viterbi_sequence, viterbi_score = crf.crf_decode(logits, transition_params, sequence_lengths_t)

        loss = tf.reduce_mean(-log_likelihood)
        return loss, viterbi_sequence, logits, viterbi_score


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # (total_loss, per_example_loss, logits, probabilities, predict) = create_model(
        (total_loss, viterbi_sequence, logits, viterbi_score) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, params["max_seq_length"], params["batch_size"])

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:  # 判断是否是初次训练，要是断点训练init_checkpoint设置为None
            (assignment_map, initialized_variable_names
             ) = bert.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = bert.optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # For mode==ModeKeys.TRAIN: 需要的参数是 loss and train_op.
                loss=total_loss,
                train_op=train_op,
            )
            return output_spec
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(total_loss_, label_ids_, viterbi_sequence_, is_real_example_):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids_, predictions=viterbi_sequence_, weights=is_real_example_)
                loss = tf.metrics.mean(values=total_loss_, weights=is_real_example_)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            # eval_metrics = (metric_fn, [per_example_loss, label_ids, predict, is_real_example])
            eval_metrics = metric_fn(total_loss, label_ids, viterbi_sequence, is_real_example)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # For mode==ModeKeys.EVAL:  需要的参数是 loss.
                loss=total_loss,
                eval_metric_ops=eval_metrics,
            )
            return output_spec
        elif mode == tf.estimator.ModeKeys.PREDICT:
            # 将实际值和预测值生成字典
            predictions = {
                "input_ids": input_ids,
                "label_ids": label_ids,
                "predicts": viterbi_sequence
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # For mode==ModeKeys.PREDICT: 需要的参数是 predictions.
                predictions=predictions,
            )
            return output_spec
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % mode)

    return model_fn

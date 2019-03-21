import tensorflow as tf
import numpy as np
import cv2


def max_pool_2d_nxn_regions(inputs, output_size: int, mode: str):
    inputs_shape = tf.shape(inputs)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)

    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))

    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=(1, 2))
            result.append(pool_result)
    return result


def spatial_pyramid_pool(inputs, dimensions, mode='max', implementation='kaiming'):
    pool_list = []
    if implementation == 'kaiming':  # kaiming = 何凱明
        for pool_dim in dimensions:
            pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, mode)
    else:
        shape = inputs.get_shape().as_list()
        for d in dimensions:
            h = shape[1]
            w = shape[2]
            ph = np.ceil(h * 1.0 / d).astype(np.int32)
            pw = np.ceil(w * 1.0 / d).astype(np.int32)
            sh = np.floor(h * 1.0 / d + 1).astype(np.int32)
            sw = np.floor(w * 1.0 / d + 1).astype(np.int32)
            pool_result = tf.nn.max_pool(inputs,
                                         ksize=[1, ph, pw, 1],
                                         strides=[1, sh, sw, 1],
                                         padding='SAME')
            pool_list.append(tf.reshape(pool_result, [tf.shape(inputs)[0], -1]))
    return tf.concat(pool_list, 1)


def read_image(path, resize=None):
    taipei = cv2.imread(path)
    if resize:
        taipei = cv2.resize(taipei, resize)
    taipei = cv2.cvtColor(taipei, cv2.COLOR_BGR2RGB)
    taipei = taipei / 255
    taipei = np.expand_dims(taipei, axis=0)
    return taipei


def main():
    """
    在 SPP 前的一層如果是 (1, None, None, 3) （即此例）
    則經過 dimensions = [4, 2, 1] 會產出
    (4*4 + 2*2 + 1*1) * 3 = 63 個特徵點 size:(1, 63)
    之後就可以接到 Dense Net 繼續做下去
    """
    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3))

    output_tensor = spatial_pyramid_pool(input_tensor, [4, 2, 1], implementation='kaiming')
    with tf.Session() as sess:
        taipei = read_image('images/taipei.jpg')
        human = read_image('images/human.jpg')

        output_result_1 = sess.run(output_tensor, feed_dict={input_tensor: taipei})
        output_result_2 = sess.run(output_tensor, feed_dict={input_tensor: human})

    print('output_result_1 shape: {}'.format(output_result_1.shape))
    print('output_result_2 shape: {}'.format(output_result_2.shape))
    assert output_result_1.shape == output_result_2.shape

    """
    如果 implementation 非 kaiming 方法，變成前一層 placeholder 必須有明確的size
    說真的，這樣似乎失去SPP 的優勢
    """
    input_tensor2 = tf.placeholder(tf.float32, shape=(1, 600, 600, 3))

    output_tensor2 = spatial_pyramid_pool(input_tensor2, [4, 2, 1], implementation='not kaiming')
    with tf.Session() as sess:
        taipei = read_image('images/taipei.jpg', resize=(600, 600))
        human = read_image('images/human.jpg', resize=(600, 600))

        output_result_1 = sess.run(output_tensor2, feed_dict={input_tensor2: taipei})
        output_result_2 = sess.run(output_tensor2, feed_dict={input_tensor2: human})

    print('output_result_1 shape: {}'.format(output_result_1.shape))
    print('output_result_2 shape: {}'.format(output_result_2.shape))
    assert output_result_1.shape == output_result_2.shape


if __name__ == '__main__':
    main()

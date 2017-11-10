import argparse
import progressbar
import cv2
import tensorflow as tf


# Monitored region of the video
XMIN = 1470
XMAX = 2840
YMIN = 50
YMAX = 2000

SIZE = 28
MIN_CONTOUR_SIZE = 45
WINDOW_SIZE = SIZE


def build_graph(size):
    # CNN
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, size, size, 1])
    y_ = tf.placeholder(tf.float32, [None, 3])

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, SIZE, SIZE, 1])

    with tf.name_scope('conv'):
        W_conv = weight_variable([3, 3, 1, 8])
        b_conv = bias_variable([8])
        h_conv = tf.nn.relu(tf.nn.conv2d(
            x_image, W_conv, strides=[1, 1, 1, 1], padding='SAME'))

    with tf.name_scope('pool'):
        h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([SIZE // 2 * SIZE // 2 * 8, 100])
        b_fc1 = bias_variable([100])

        h_pool_flat = tf.reshape(h_pool, [-1, SIZE // 2 * SIZE // 2 * 8])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([100, 3])
        b_fc2 = bias_variable([3])

        y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return x, y


def find_centroids(thresh, min_size):
    if min_size < 1:
        raise ValueError('min_size must be at least 1')

    _, cont, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    M = map(cv2.moments, cont)

    cxs = []
    cys = []

    for i, m in enumerate(M):
        if m['m00'] < min_size:
            continue

        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

        cxs.append(cx)
        cys.append(cy)

    return cxs, cys


def get_masked_window(grayed, cx, cy, size):
    windowed = grayed[cy - size // 2:cy +
                      size // 2, cx - size // 2:cx + size // 2]
    _, bw = cv2.threshold(
        windowed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cont, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if 0 < len(cont) < 3:
        return windowed * (bw == 0)
    else:
        return None


def main(args):
    # Load input video.
    input_video = cv2.VideoCapture(args.input)

    # Define video resolution and fps.
    video_width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not video_width:
        video_width = args.resolution[0]
    if not video_height:
        video_height = args.resolution[1]
    if not (video_width or video_height):
        raise argparse.ArgumentError('--resolution', 'resolution not defined')

    if args.fps:
        video_fps = args.fps
    else:
        video_fps = input_video.get(cv2.CAP_PROP_FPS)
    if not video_fps:
        raise argparse.ArgumentError('--fps', 'fps not defined')

    # Open output video
    output_video = cv2.VideoWriter(args.output,
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   video_fps,
                                   (video_width, video_height))

    # Define input video length in seconds.
    video_end = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps)
    if args.frame_range[1]:
        video_end = int(args.frame_range[1] / video_fps)
    if not video_end:
        video_end = 5 * 3600  # Default max duration 5 hours.

    video_start = 0
    if args.frame_range[0]:
        video_start = int(args.frame_range[0] / video_fps)

    video_duration = video_end - video_start

    # Define monitored region.
    if all(args.region):
        xmin, xmax, ymin, ymax = args.region
    else:
        xmin, xmax, ymin, ymax = 0, video_width, 0, video_height

    # Progress bar.
    progressbar.streams.wrap_stderr()
    bar = progressbar.ProgressBar(max_value=(video_duration),
                                  redirect_stdout=True)
    bar.update(0)

    # Load TF model.
    x, y = build_graph(SIZE)
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, args.checkpoint)

    # Tracking
    frame_count = 0
    video_time = 0
    while input_video.isOpened() and video_time < video_end:
        _, frame = input_video.read()

        cropped = frame[ymin:ymax, xmin:xmax]
        grayed = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(grayed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        cxs, cys = find_centroids(thresh, MIN_CONTOUR_SIZE)

        masks = []
        positions = []

        for cx, cy in zip(cxs, cys):
            masked = get_masked_window(grayed, cx, cy, WINDOW_SIZE)
            if masked is not None and masked.size == WINDOW_SIZE * WINDOW_SIZE:
                masks.append(masked)
                positions.append((cx, cy))

        # Feed masked windows to trained model for prediction.
        predictions = sess.run(tf.argmax(y, 1), feed_dict={x: masks})
        for prediction, (cx, cy) in zip(predictions, positions):
            if prediction == 1:
                cv2.rectangle(frame, (cx + XMIN - 15, cy + YMIN - 15),
                              (cx + XMIN + 15, cy + YMIN + 15), (0, 255, 0), 3)
            # elif prediction == 2:
            #    cv2.rectangle(frame, (cx + XMIN - 15, cy + YMIN - 15),
            #                  (cx + XMIN + 15, cy + YMIN + 15), (255, 0, 0), 3)

        output_video.write(frame)
        # Update progress bar
        frame_count += 1
        if frame_count % 30 == 0:
            video_time += 1
            bar.update(video_time)

    output_video.release()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str,
                        default='../data/videos/TandemRun.mp4',
                        help='path to input video file')
    parser.add_argument('--output', type=str,
                        default='../data/videos/tracked.mp4',
                        help='path to output video file')
    parser.add_argument('--resolution', nargs=2, type=int,
                        default=[None, None],
                        help='resolution of input and output videos')
    parser.add_argument('--fps', type=int,
                        default=None,
                        help='frame per second of output videos')
    parser.add_argument('--frame_range', nargs=2, type=int,
                        default=[None, None],
                        help='frame range of input video')
    parser.add_argument('--region', nargs=4, type=int,
                        default=[XMIN, YMIN, XMAX, YMAX],
                        help='region of input video to be monitored')
    parser.add_argument('--tfcheckpoint', type=str,
                        default='../data/tfsave/trained_model.ckpt',
                        help='path to TensorFlow checkpoint')

    args = parser.parse_args()

    main(args)

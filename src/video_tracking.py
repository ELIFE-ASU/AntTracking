import os
import argparse
import progressbar
import cv2
import numpy as np
import tensorflow as tf


SIZE = 28
MIN_CONTOUR_SIZE = 40
MAX_CONTOUR_SIZE = 250
LABEL_SIZE = 40


def get_parameters(input_video, args):
    # Define video resolution and fps.
    video_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not video_width:
        video_width = args.resolution[0]
    if not video_height:
        video_height = args.resolution[1]
    if not (video_width or video_height):
        raise argparse.ArgumentError('--resolution', 'resolution not defined')

    if args.fps:
        video_fps = args.fps
    else:
        video_fps = int(input_video.get(cv2.CAP_PROP_FPS))
    if not video_fps:
        raise argparse.ArgumentError('--fps', 'fps not defined')

    # Define input video length in seconds.
    video_end = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps)
    if args.time[1]:
        video_end = args.time[1]
    if not video_end:
        video_end = 5 * 3600  # Default max duration 5 hours.

    video_start = 0
    if args.time[0]:
        video_start = args.time[0]

    # Define monitored region.
    if any([val is None for val in args.region]):
        xmin, ymin, xmax, ymax = 0, 0, video_width, video_height
    else:
        xmin, ymin, xmax, ymax = args.region

    return {'resolution': (video_width, video_height, video_fps),
            'time': (video_start, video_end),
            'region': (xmin, ymin, xmax, ymax)}


def build_graph(size):
    # CNN
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, size, size])

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


def find_centroids(bw, min_size, max_size):
    if min_size < 1:
        raise ValueError('min_size must be at least 1')

    _, cont, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    M = map(cv2.moments, cont)

    centroids = []

    for m in M:
        if m['m00'] < min_size or m['m00'] > max_size:
            continue

        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

        centroids.append((cx, cy))

    return centroids


def get_masked_window(grayed, cx, cy, size):
    ymin = cy - size // 2 if cy - size // 2 > 0 else 0
    xmin = cx - size // 2 if cx - size // 2 > 0 else 0
    windowed = grayed[ymin:ymin + size, xmin:xmin + size]
    th, bw = cv2.threshold(
        windowed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cont, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if 0 < len(cont) < 3:
        return windowed * (bw == 0), th  # / 256.  # Normalize.
    return None, None


def mesh_positions(positions, size):
    new_positions = []
    positions = list(positions)
    while positions:
        cx, cy = positions.pop()
        i = 0
        n = 1
        sum_x, sum_y = cx, cy
        while i < len(positions):
            cx_, cy_ = positions[i]
            if abs(cx - cx_) < size and abs(cy - cy_) < size:
                sum_x += cx_
                sum_y += cy_
                n += 1
                positions.pop(i)
            else:
                i += 1
        new_positions.append((int(sum_x / n), int(sum_y / n)))
    return new_positions


def locate_tandem(frame, region, classifier):
    xmin, ymin, xmax, ymax = region
    sess = classifier['session']
    x = classifier['x']
    y = classifier['y']

    cropped = frame[ymin:ymax, xmin:xmax]
    grayed = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(grayed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    centroids = find_centroids(bw, MIN_CONTOUR_SIZE, MAX_CONTOUR_SIZE)

    masks = []
    positions = []

    for cx, cy in centroids:
        masked, _ = get_masked_window(grayed, cx, cy, SIZE)
        if masked is not None and masked.size == SIZE * SIZE:
            masks.append(masked)
            positions.append((cx + xmin, cy + ymin))  # Absolute positions.
    positions = np.asarray(positions)

    # Feed masked windows to trained model for prediction.
    predictions = sess.run(tf.argmax(y, 1), feed_dict={x: masks})

    # Put tags on video according to the predictions.
    tandem_positions = positions[predictions == 1]
    tandem_positions = mesh_positions(tandem_positions, SIZE)

    return tandem_positions


def individual_positions(frame, tandem_position, window_size):
    global counter
    xmin, xmax = tandem_position[0] - \
        window_size // 2, tandem_position[0] + window_size // 2
    ymin, ymax = tandem_position[1] - \
        window_size // 2, tandem_position[1] + window_size // 2

    window = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    th, bw = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    centroids = find_centroids(bw, MIN_CONTOUR_SIZE, MAX_CONTOUR_SIZE)

    while len(centroids) < 2 and th > 80:
        th, bw = cv2.threshold(window, th - 5, 255, cv2.THRESH_BINARY)
        centroids = find_centroids(bw, 10, MAX_CONTOUR_SIZE)

    return [(cx + xmin, cy + ymin) for cx, cy in centroids]


def tag_tandem(frame, tandem_positions, window_size):
    for cx, cy in tandem_positions:
        ants_positions = individual_positions(frame, (cx, cy), window_size)
        cv2.rectangle(frame, (cx - window_size // 2, cy - window_size // 2),
                      (cx + window_size // 2, cy + window_size // 2), (0, 255, 0), 3)

        # Mark the center of mass of each ant.
        for antx, anty in ants_positions:
            cv2.rectangle(frame, (antx - 2, anty - 2),
                          (antx + 2, anty + 2), (0, 0, 255), 2)
    return frame


def main(args):
    # Load input video.
    input_video = cv2.VideoCapture(args.input)
    if not input_video.isOpened():
        raise IOError('video "{}" not open'.format(args.input))

    # Get parameters.
    parameters = get_parameters(input_video, args)
    video_width, video_height, video_fps = parameters['resolution']
    video_start, video_end = parameters['time']
    xmin, ymin, xmax, ymax = parameters['region']
    video_duration = video_end - video_start

    # Locating user specified video start.
    print('Seeking specified video start...', end='')
    frame_count = 0
    while frame_count < video_start * video_fps:
        ret, frame = input_video.read()
        if not ret:
            print('Input video shorter than {}s'.format(video_start))
            input_video.release()
            return
        frame_count += 1
    print(' done.')

    # Open output video for recording.
    if os.path.exists(args.output):
        os.remove(args.output)

    output_video = cv2.VideoWriter(args.output,
                                   cv2.VideoWriter_fourcc(*args.codec),
                                   video_fps,
                                   (video_width, video_height))

    # Load TF model.
    x, y = build_graph(SIZE)
    saver = tf.train.Saver()

    print('Loading classifier model...', end='')
    sess = tf.Session()
    saver.restore(sess, args.checkpoint)

    classifier = {'session': sess, 'x': x, 'y': y}
    print(' done.')

    print('Tracking...')
    # Progress bar.
    progressbar.streams.wrap_stderr()
    bar = progressbar.ProgressBar(max_value=(video_duration),
                                  redirect_stdout=True)
    bar.update(0)

    # Tracking.
    video_time = 0
    while video_time < video_duration:
        ret, frame = input_video.read()
        if not ret:
            bar.update(video_time)
            break

        tandem_positions = locate_tandem(
            frame, (xmin, ymin, xmax, ymax), classifier)

        tagged_frame = tag_tandem(frame, tandem_positions, args.label_size)

        output_video.write(tagged_frame)
        # Update progress bar
        frame_count += 1
        if frame_count % 30 == 0:
            video_time += 1
            bar.update(video_time)

    sess.close()
    input_video.release()
    output_video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str,
                        help='path to input video file')
    parser.add_argument('--output', '-o', type=str,
                        help='path to output video file')
    parser.add_argument('--resolution', nargs=2, type=int,
                        help='resolution of input and output videos')
    parser.add_argument('--fps', type=int,
                        help='frame per second of output videos')
    parser.add_argument('--time', '-t', nargs=2, type=int,
                        help='time range of input video in seconds')
    parser.add_argument('--region', nargs=4, type=int,
                        help='region of input video to be monitored')
    parser.add_argument('--checkpoint', type=str,
                        default='../data/tf_save/trained_model_v1/trained_model.ckpt',
                        help='path to TensorFlow checkpoint')
    parser.add_argument('--label_size', type=int,
                        default=LABEL_SIZE,
                        help='size of label box')
    parser.add_argument('-c', '--codec', type=str,
                        default='X264',
                        help='encoding codec for output video')

    main(parser.parse_args())

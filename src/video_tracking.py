from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import json
import progressbar
import cv2
import numpy as np
import tensorflow as tf


SIZE = 28
MIN_CONTOUR_SIZE = 40
MAX_CONTOUR_SIZE = 250
LABEL_SIZE = 40


def get_parameters(input_video, args):
    """Parse arguments and determine parameters of input video."""
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
    if args.time[1]:
        video_end = args.time[1]
    else:
        try:
            video_end = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps)
        except TypeError:
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
    """Build a TensorFlow graph as the classifier."""
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
    """Find centroids of all blobs."""
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
    """Return candidate window to be tested by classifier."""
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
    """Combine positions that are closer than size into one position."""
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
    """Locate all tandems in specified region of a frame."""
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

    # Collect tandem positions.
    tandem_positions = positions[predictions == 1]
    tandem_positions = mesh_positions(tandem_positions, SIZE)

    return tandem_positions


def individual_positions(frame, tandem_position, window_size):
    """Find positions of individual ants of a tandem pair."""
    xmin, xmax = tandem_position[0] - \
        window_size // 2, tandem_position[0] + window_size // 2
    ymin, ymax = tandem_position[1] - \
        window_size // 2, tandem_position[1] + window_size // 2

    window = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)

    th = 105
    centroids = []

    while len(centroids) < 2 and th > 80:
        th, bw = cv2.threshold(window, th - 5, 255, cv2.THRESH_BINARY)
        centroids = find_centroids(bw, 10, MAX_CONTOUR_SIZE)

    return [(cx + xmin, cy + ymin) for cx, cy in centroids]


def tandem_ants(frame, region, classifier, window_size):
    """Find tandem pairs and positions of individual ants in the tandems."""
    tandem_candidates = locate_tandem(frame, region, classifier)

    tandems = []
    ants = []
    for cx, cy in tandem_candidates:
        ants_candidates = individual_positions(frame, (cx, cy), window_size)

        if len(ants_candidates) == 2:
            tandems.append((cx, cy))
            ants.append(ants_candidates)

    return tandems, ants


def match_order(positions1, positions2):
    """Match the order of positions2 with positions1 to ensure continuity."""
    def distance2(xy1, xy2):
        return (xy1[0] - xy2[0]) * (xy1[0] - xy2[0]) + (xy1[1] - xy2[1]) * (xy1[1] - xy2[1])

    d11 = distance2(positions1[0], positions2[0])
    d12 = distance2(positions1[0], positions2[1])
    d21 = distance2(positions1[1], positions2[0])
    d22 = distance2(positions1[1], positions2[1])

    if d12 < d22 and d11 > d21:
        return [positions2[1], positions2[0]]
    return positions2


def gather(tandems, ants, tandems_history, video_time, frame_num, window_size):
    """Collect information of tandem runners to corresponding tracks according to history."""
    latency = 5
    labels = []
    for tandem, pairs in zip(tandems, ants):
        for label, info in tandems_history.items():
            anchor_x, anchor_y = info['last_seen']['position']
            anchor_t = info['last_seen']['time']
            # Attach to positions to existing tandems.
            if (video_time - anchor_t < latency
                    and abs(tandem[0] - anchor_x) < window_size
                    and abs(tandem[1] - anchor_y) < window_size):
                info['last_seen']['time'] = video_time
                info['last_seen']['frame'] = frame_num
                info['last_seen']['position'] = tandem
                # Check continuity.
                ordered_pairs = match_order(info['ants_positions'][-1], pairs)
                info['ants_positions'].append(ordered_pairs)
                labels.append(label)
                break
        # Otherwise create a new track.
        else:
            num_labels = len(tandems_history)
            tandems_history[num_labels] = {
                'last_seen':
                {
                    'position': tandem,
                    'time': video_time,
                    'frame': frame_num
                },
                'ants_positions': [pairs]
            }
            labels.append(num_labels)

    return labels


def redeem_lost(frame, tandems_history, frame_num, window_size):
    """
    Try to locate individual ants where the tandem was last seen, if classifier
    loses track in a frame.
    """
    latency = 30 * 3
    for info in tandems_history.values():
        if frame_num - latency < info['last_seen']['frame'] < frame_num:
            ants = individual_positions(
                frame, info['last_seen']['position'], window_size)
            if len(ants) == 2:
                info['last_seen']['frame'] = frame_num
                ordered_pairs = match_order(info['ants_positions'][-1], ants)
                info['ants_positions'].append(ordered_pairs)

    tandems, labels, ants = [], [], []
    for label, info in tandems_history.items():
        if info['last_seen']['frame'] == frame_num:
            labels.append(label)
            tandems.append(info['last_seen']['position'])
            ants.append(info['ants_positions'][-1])

    return labels, tandems, ants


def tag_tandem(frame, tandems, ants, labels, window_size):
    """Tag the tandems in the output video frame."""
    for (cx, cy), pairs, label in zip(tandems, ants, labels):
        cv2.rectangle(frame, (cx - window_size // 2, cy - window_size // 2),
                      (cx + window_size // 2, cy + window_size // 2), (0, 255, 0), 3)
        cv2.putText(frame, str(label), (cx - 8, cy - window_size + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Mark the center of mass of each ant.
        if len(pairs) == 2:
            (antx1, anty1), (antx2, anty2) = pairs

            displacement = (antx1 - antx2, anty1 - anty2)
            distance = np.linalg.norm(displacement)

            cv2.rectangle(frame, (antx1 - 1, anty1 - 1), (antx1 + 1, anty1 + 1),
                          (0, 0, 255), 2)

            text1x = int(displacement[0] * 20 / distance) + antx1 - 8
            text1y = int(displacement[1] * 20 / distance) + anty1 + 8
            cv2.putText(frame, str(0), (text1x, text1y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (antx2 - 1, anty2 - 1), (antx2 + 1, anty2 + 1),
                          (0, 255, 255), 2)

            text2x = int(-displacement[0] * 20 / distance) + antx2 - 8
            text2y = int(-displacement[1] * 20 / distance) + anty2 + 8
            cv2.putText(frame, str(1), (text2x, text2y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
    frame_num = 0
    while frame_num < video_start * video_fps:
        ret, frame = input_video.read()
        if not ret:
            print('Input video shorter than {}s'.format(video_start))
            input_video.release()
            return
        frame_num += 1
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
    tandems_history = {}
    video_time = 0
    while video_time < video_duration:
        ret, frame = input_video.read()
        if not ret:
            bar.update(video_time)
            break
        # Find the position of tandems and their individual ants.
        tandems, ants = tandem_ants(
            frame, (xmin, ymin, xmax, ymax), classifier, args.label_size)
        # Record the postions in tandems_history, matching the labels.
        labels = gather(tandems, ants, tandems_history,
                        video_time, frame_num, args.label_size)
        labels, tandems, ants = redeem_lost(
            frame, tandems_history, frame_num, args.label_size)
        # Collect all tandems of the current frame.

        # Tag the tandems and ants.
        tagged_frame = tag_tandem(
            frame, tandems, ants, labels, args.label_size)

        output_video.write(tagged_frame)
        # Update progress bar
        frame_num += 1
        if frame_num % 30 == 0:
            video_time += 1
            bar.update(video_time)

    sess.close()
    input_video.release()
    output_video.release()

    with open(args.log, 'w') as log_json:
        json.dump(tandems_history, log_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str,
                        help='path to input video file')
    parser.add_argument('--output', '-o', type=str,
                        help='path to output video file')
    parser.add_argument('--log', '-l', type=str,
                        help='path to the json log file')
    parser.add_argument('--resolution', nargs=2, type=int,
                        help='resolution of input and output videos')
    parser.add_argument('--fps', type=int,
                        help='frame per second of output videos')
    parser.add_argument('--time', '-t', nargs=2, type=int,
                        default=[None, None],
                        help='time range of input video in seconds')
    parser.add_argument('--region', nargs=4, type=int,
                        default=[None, None, None, None],
                        help='region of input video to be monitored')
    parser.add_argument('--checkpoint', type=str,
                        default='../data/tf_save/trained_model_v2/model.ckpt',
                        help='path to TensorFlow checkpoint')
    parser.add_argument('--label_size', type=int,
                        default=LABEL_SIZE,
                        help='size of label box')
    parser.add_argument('-c', '--codec', type=str,
                        default='X264',
                        help='encoding codec for output video')

    main(parser.parse_args())

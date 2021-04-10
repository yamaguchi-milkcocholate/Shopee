import tensorflow as tf


# Function to decode our images
def decode_image(image_data, image_size):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example, image_size):
    LABELED_TFREC_FORMAT = {
        "posting_id": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label_group": tf.io.FixedLenFeature([], tf.int64),
        "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['posting_id']
    image = decode_image(example['image'], image_size=image_size)
    label_group = tf.cast(example['label_group'], tf.int32)
    matches = example['matches']
    return posting_id, image, label_group, matches

# This function loads TF Records and parse them into tensors
def load_dataset(filenames, tf_expt, image_size, ordered = False):
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = tf_expt)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda x: read_labeled_tfrecord(x, image_size=image_size), num_parallel_calls = tf_expt) 
    return dataset


def arcface_format(posting_id, image, label_group, matches):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group, matches


# This function is to get our training tensors
def get_training_dataset(filenames, config, data_augment, ordered = False):
    dataset = load_dataset(filenames, config.tf_expt, config.image_size, ordered = ordered)
    dataset = dataset.map(data_augment, num_parallel_calls = config.tf_expt)
    dataset = dataset.map(arcface_format, num_parallel_calls = config.tf_expt)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048, seed=config.seed)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(config.tf_expt)
    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, config, data_augment, ordered = True):
    dataset = load_dataset(filenames, config.tf_expt, config.image_size, ordered = ordered)
    dataset = dataset.map(arcface_format, num_parallel_calls = config.tf_expt)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(config.tf_expt) 
    return dataset



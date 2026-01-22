import tensorflow as tf

from keras.applications.mobilenet_v2 import preprocess_input


class HAMDataGenerator:
    def __init__(self, train_df):
        self.size = [224, 224]
        self.batch_size = 16
        self.classes = train_df.dx.cat.categories.astype(str).tolist()
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(20 / 360),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="augmentation",
        )

    def _load_and_preprocess(self, filename, label):
        image = tf.io.read_file(filename=filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, size=self.size)
        image = preprocess_input(image)
        label = tf.one_hot(label, depth=len(self.classes))

        return image, label

    def _apply_augmentation(self, image, label):
        image = self.data_augmentation(image, training=True)
        return image, label

    def flow_from_dataframe(self, df, directory, x_col, y_col, shuffle=False):
        filepaths = [f"{directory}/{img}" for img in df[x_col]]
        labels = df[y_col].values
        label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        label_indices = [label_to_index[label] for label in labels]
        ds = tf.data.Dataset.from_tensor_slices((filepaths, label_indices))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(filepaths))

        ds = ds.map(self._load_and_preprocess)
        ds = ds.batch(batch_size=self.batch_size)
        return ds.map(self._apply_augmentation)

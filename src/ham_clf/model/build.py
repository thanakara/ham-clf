import tensorflow as tf

from omegaconf import DictConfig
from keras.applications import MobileNetV2


class MobileNetV2FT0:
    def __init__(self, config: DictConfig):
        self.config = config
        self.base_model = MobileNetV2(
            input_shape=self.config.training.transfer.input_shape,
            include_top=False,
            weights=self.config.training.transfer.weights,
            alpha=self.config.training.transfer.alpha,
        )

    def _stage_one(self):
        self.base_model.trainable = False

        X = self.base_model.output
        X = tf.keras.layers.GlobalAveragePooling2D()(X)
        X = tf.keras.layers.Dense(self.config.training.transfer.units, activation="relu")(X)
        X = tf.keras.layers.Dropout(rate=self.config.training.transfer.dropout_rate)(X)
        output = tf.keras.layers.Dense(self.config.training.transfer.n_classes, activation="softmax")(X)
        model = tf.keras.Model(inputs=[self.base_model.input], outputs=[output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.training.transfer.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return model

    def transfer(self, train_ds, valid_ds, class_weight_dict, callbacks):
        model = self._stage_one()
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=self.config.training.transfer.epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks,
        )
        self.model = model
        return history

    def _stage_two(self):
        self.base_model.trainable = True
        for layer in self.base_model.layers[: -self.config.training.fine_tuning.n_last_layers]:
            layer.trainable = False
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.training.fine_tuning.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return self.model

    def fine_tune(self, train_ds, valid_ds, class_weight_dict, callbacks):
        model = self._stage_two()
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=self.config.training.fine_tuning.epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks,
        )
        self.model = model
        self.model.save(self.config.training.fine_tuning.modelname)
        return history

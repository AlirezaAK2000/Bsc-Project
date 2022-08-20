import numpy as np
import tensorflow as tf
import time
from functools import reduce
from IPython import display
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime


class Conv2DMaxpoolBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size=3,
                 padding='valid',
                 pool_size=(2, 2),
                 activation='relu') -> None:
        super().__init__()
        self.conv_in = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation)
        self.batchNorm1 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=activation)

        self.conv_res = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation)

        self.batchNorm2 = tf.keras.layers.BatchNormalization()

        self.maxPool = tf.keras.layers.MaxPool2D(pool_size)
        # self.conv_out = tf.keras.layers.Conv2D(
        #     filters=filters,
        #     kernel_size=4,
        #     strides=2,
        #     activation=activation)

        # self.batchNorm3 = tf.keras.layers.BatchNormalization(
        #     momentum=.999, epsilon=1e-5)

    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.batchNorm1(x)

        x = self.conv1(x)
        res = self.conv_res(inputs)
        x = tf.math.add(x, res)
        x = self.batchNorm2(x)

        x = self.maxPool(x)

        # x = self.conv_out(x)
        # x = self.batchNorm3(x)

        return x


class Conv2DTransposeUpSamplingBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size=3,
                 padding='valid',
                 pool_size=(2, 2),
                 activation='relu'):
        super().__init__()
        self.conv_in = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation)
        self.batchNorm1 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=activation)

        self.conv_res = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation)

        self.batchNorm2 = tf.keras.layers.BatchNormalization()

        self.maxPool = tf.keras.layers.UpSampling2D(pool_size)
        # self.conv_out = tf.keras.layers.Conv2DTranspose(
        #     filters=filters,
        #     kernel_size=4,
        #     strides=2,
        #     activation=activation)

        # self.batchNorm3 = tf.keras.layers.BatchNormalization(
        #     momentum=.999, epsilon=1e-5)

    def call(self, inputs):

        x = self.conv_in(inputs)
        x = self.batchNorm1(x)

        x = self.conv1(x)
        res = self.conv_res(inputs)
        x = tf.math.add(x, res)
        x = self.batchNorm2(x)

        x = self.maxPool(x)

        # x = self.conv_out(x)
        # x = self.batchNorm3(x)

        return x


class VariationalAutoEncoder(tf.keras.Model):

    def __init__(self, latent_dim, input_shape=(512, 512, 3)) -> None:
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                Conv2DMaxpoolBlock(filters=16, kernel_size=5),

                Conv2DMaxpoolBlock(filters=32),

                Conv2DMaxpoolBlock(filters=64),

                Conv2DMaxpoolBlock(filters=64),

                Conv2DMaxpoolBlock(filters=64),

                Conv2DMaxpoolBlock(filters=64),

                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        encoder_shape = list(
            filter(lambda layer: 'conv2d_maxpool_block' in layer.name, self.encoder.layers))[-1].output_shape[1:]

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim)),
                tf.keras.layers.Dense(
                    reduce(lambda x, y: x*y, encoder_shape), activation='relu'),
                tf.keras.layers.Reshape(target_shape=encoder_shape),

                Conv2DTransposeUpSamplingBlock(filters=64, padding='same'),

                Conv2DTransposeUpSamplingBlock(filters=64),

                Conv2DTransposeUpSamplingBlock(filters=64),

                Conv2DTransposeUpSamplingBlock(filters=64),

                Conv2DTransposeUpSamplingBlock(filters=32),

                Conv2DTransposeUpSamplingBlock(
                    filters=16, kernel_size=5, activation='linear'),

                # tf.keras.layers.Conv2DTranspose(
                #     filters=3, kernel_size=3, padding = 'same'),

            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class LowResolutionVariationalAutoEncoder(tf.keras.Model):

    def __init__(self, latent_dim, input_shape=(256, 256, 3)) -> None:
        super(LowResolutionVariationalAutoEncoder, self).__init__()

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                Conv2DMaxpoolBlock(filters=16, kernel_size=5),

                Conv2DMaxpoolBlock(filters=32),

                Conv2DMaxpoolBlock(filters=64),

                Conv2DMaxpoolBlock(filters=64),

                Conv2DMaxpoolBlock(filters=64),

                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        encoder_shape = list(
            filter(lambda layer: 'conv2d_maxpool_block' in layer.name, self.encoder.layers))[-1].output_shape[1:]

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim)),
                tf.keras.layers.Dense(
                    reduce(lambda x, y: x*y, encoder_shape), activation='relu'),
                tf.keras.layers.Reshape(target_shape=encoder_shape),

                Conv2DTransposeUpSamplingBlock(filters=64, padding='same'),

                Conv2DTransposeUpSamplingBlock(filters=64),

                Conv2DTransposeUpSamplingBlock(filters=64),

                Conv2DTransposeUpSamplingBlock(filters=32),

                Conv2DTransposeUpSamplingBlock(filters=16, kernel_size=5),

                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, padding='same'),

            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('progress/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train_model(model, train_dataset, test_dataset, epochs, optimizer, test_sample, save_model=True, check_point=None):

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") if check_point == None else check_point
    train_log_dir = 'logs/' + current_time + '/train'
    test_log_dir = 'logs/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    losses = []
    generate_and_save_images(model, 0, test_sample)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        with tqdm(train_dataset) as tepoch:
            for batch_index in range(len(train_dataset)):
                train_x, _ = train_dataset[batch_index]
                tepoch.set_description(f"Epoch {epoch}")
                loss_amount = train_step(model, train_x, optimizer)
                train_loss(loss_amount)
                assert not tf.math.is_nan(train_loss.result())
                tepoch.set_postfix(ELBO=-train_loss.result())
                tepoch.update(1)
                with train_summary_writer.as_default():
                    tf.summary.scalar('ELBO', -train_loss.result(), step=epoch)

        end_time = time.time()
        with tqdm(test_dataset) as tepoch:
            for batch_index in range(len(test_dataset)):
                test_x, _ = test_dataset[batch_index]
                tepoch.set_description(f"Epoch {epoch} test batches")
                test_loss(compute_loss(model, test_x))
                with test_summary_writer.as_default():
                    tf.summary.scalar('ELBO', -test_loss.result(), step=epoch)
                tepoch.update(1)
        elbo = -test_loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        generate_and_save_images(model, epoch, test_sample)
        losses.append((-test_loss.result(), -train_loss.result()))

        train_loss.reset_states()
        test_loss.reset_states()

        if epoch % 2 == 0 and save_model:
            model.save_weights(f"checkpoints/vae{current_time}/checkpoint")
            print("model saved!!!")

    return losses

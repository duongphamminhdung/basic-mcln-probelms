from tqdm import tqdm
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras import layers as nn
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation as Activ

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

seed = 256
tf.random.set_seed(seed)

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        
        #drop out
        self.drop_out = nn.Dropout(0.2)
        #CNN layers
        self.conv1 = nn.Conv2D(input_shape=(1, 28, 28), filters=3, kernel_size = (3, 3), strides=(1, 1), padding='same')
        #in 1x28x28 out 3x28x28
        
        self.conv2 = nn.Conv2D(input_shape=(3, 28, 28), filters=9, kernel_size= (3, 3), strides=(1, 1), padding='same')
        self.act = nn.ReLU()
        self.pool2 = nn.MaxPool2D(pool_size=(2, 2))
        #in 3x28x28 out 9x14x14
        
        self.flat = nn.Flatten()
        #int 1x1764 out


        #linear layers
        self.linear1 = nn.Dense(256, activation='sigmoid')
        # self.actfunc1 = Activ(tf.nn.Sigmoid)
        self.linear2 = nn.Dense(10)
        self.softmax = nn.Softmax(axis=1)

    def call(self, x):
        x = (self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        # import ipdb;ipdb.set_trace()
        x = self.drop_out(self.flat(x))
        x = self.linear1(x)
        x = self.linear2(self.act(x))
        x = self.softmax(x)
        
        return x

# Create an instance of the model
model = MnistModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.experimental.SGD(learning_rate = 0.15, momentum = 0.9, nesterov=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

tf.config.run_functions_eagerly(True)
tf.debugging.set_log_device_placement(True)

EPOCHS = 200
print('optimizer:', optimizer.__class__.__name__)
for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in tqdm(train_ds):
    
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result():.4f}, '
    f'Accuracy: {train_accuracy.result() * 100:.2f}, '
    f'Test Loss: {test_loss.result():.4f}, '
    f'Test Accuracy: {test_accuracy.result() * 100:.2f}'
  )

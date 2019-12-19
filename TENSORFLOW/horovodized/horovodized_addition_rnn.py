
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
from horovodizer_helper import *
import horovod.keras as hvd
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
from tensorflow.keras.optimizers import get as get_optimizer_by_name
hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

class CharacterTable(object):

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict(((c, i) for (i, c) in enumerate(self.chars)))
        self.indices_char = dict(((i, c) for (i, c) in enumerate(self.chars)))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for (i, c) in enumerate(C):
            x[(i, self.char_indices[c])] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=(- 1))
        return ''.join((self.indices_char[x] for x in x))

class colors():
    ok = '\x1b[92m'
    fail = '\x1b[91m'
    close = '\x1b[0m'
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True
MAXLEN = ((DIGITS + 1) + DIGITS)
chars = '0123456789+ '
ctable = CharacterTable(chars)
questions = []
expected = []
seen = set()
print('Generating data...')
while (len(questions) < TRAINING_SIZE):
    f = (lambda : int(''.join((np.random.choice(list('0123456789')) for i in range(np.random.randint(1, (DIGITS + 1)))))))
    (a, b) = (f(), f())
    key = tuple(sorted((a, b)))
    if (key in seen):
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = (q + (' ' * (MAXLEN - len(q))))
    ans = str((a + b))
    ans += (' ' * ((DIGITS + 1) - len(ans)))
    if REVERSE:
        query = query[::(- 1)]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))
print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), (DIGITS + 1), len(chars)), dtype=np.bool)
for (i, sentence) in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for (i, sentence) in enumerate(expected):
    y[i] = ctable.encode(sentence, (DIGITS + 1))
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
split_at = (len(x) - (len(x) // 10))
(x_train, x_val) = (x[:split_at], x[split_at:])
(y_train, y_val) = (y[:split_at], y[split_at:])
print('Training Data:')
print(x_train.shape)
print(y_train.shape)
print('Validation Data:')
print(x_val.shape)
print(y_val.shape)
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
print('Build model...')
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector((DIGITS + 1)))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=adapt_optimizer('adam'), metrics=['accuracy'])
model.summary()
for iteration in range(1, 200):
    print()
    print(('-' * 50))
    print('Iteration', iteration)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=adapt_epochs(1), validation_data=(x_val, y_val), callbacks=adapt_callbacks([], True), verbose=(1 if (hvd.rank() == 0) else 0))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        (rowx, rowy) = (x_val[np.array([ind])], y_val[np.array([ind])])
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', (q[::(- 1)] if REVERSE else q), end=' ')
        print('T', correct, end=' ')
        if (correct == guess):
            print(((colors.ok + '☑') + colors.close), end=' ')
        else:
            print(((colors.fail + '☒') + colors.close), end=' ')
        print(guess)

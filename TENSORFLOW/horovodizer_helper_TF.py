import horovod.tensorflow.keras as hvd
import math
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import get as get_optimizer_by_name

def adapt_optimizer(opt):
    if ('str' == opt.__class__.__name__):
        opt = get_optimizer_by_name(opt)
    opt_config = opt.get_config()
    try:
        opt_config['learning_rate'] *= hvd.size()
    except KeyError:
        opt_config['lr'] *= hvd.size()
    return hvd.DistributedOptimizer(opt.from_config(opt_config))

def adapt_callbacks(callbacks, save_checkpoints):
    hvd_callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), hvd.callbacks.MetricAverageCallback(), hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1), keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)]
    if ((hvd.rank() == 0) and save_checkpoints):
        callbacks.append(ModelCheckpoint('./checkpoint-{epoch}.h5'))
    return (hvd_callbacks + callbacks)

def adapt_epochs(epochs):
    return max(1, math.ceil((epochs // hvd.size())))

def adapt_steps(steps):
    return max(1, math.ceil((steps // hvd.size())))

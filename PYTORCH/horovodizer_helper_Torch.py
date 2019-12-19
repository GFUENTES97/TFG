import horovod.torch as hvd
import math
import torch
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import get as get_optimizer_by_name

def adapt_optimizer(optimizer, model):
    optimizer.param_groups[0]['lr'] *= hvd.size()
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def hvd_metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def adapt_callbacks(callbacks, save_checkpoints):
    hvd_callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0), hvd.callbacks.MetricAverageCallback(), hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1), keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)]
    if ((hvd.rank() == 0) and save_checkpoints):
        callbacks.append(ModelCheckpoint('./checkpoint-{epoch}.h5'))
    return (hvd_callbacks + callbacks)

def adapt_epochs(epochs):
    return max(1, math.ceil((epochs // hvd.size())))

def adapt_steps(steps):
    return max(1, math.ceil((steps // hvd.size())))

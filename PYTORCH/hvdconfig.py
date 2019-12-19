from ast import *

class TF:

    ## Keras model names:
    possible_model_names_keras=[
    'Model',
    'Sequential',
    'Xception',
    'VGG16',
    'VGG19',
    'ResNet50',
    'ResNet50V2',
    'ResNet101',
    'ResNet101V2',
    'ResNet152',
    'ResNet152V2',
    'InceptionV3',
    'InceptionResNetV2',
    'MobileNet',
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
    'NASNetLarge',
    'NASNetMobile',
    'MobileNetV2'
]

    ## IMPORTS:
    imports = [
        ImportFrom(module='horovodizer_helper_TF',names=[alias(name='*',asname=None)],level=0),
        Import(names=[alias(name='horovod.keras',asname='hvd')]),
        Import(names=[alias(name='math',asname=None)]),
        ImportFrom(module='tensorflow.keras.callbacks',names=[alias(name='ModelCheckpoint',asname=None)],level=0),
        ImportFrom(module='keras',names=[alias(name='backend',asname='K')],level=0),
        ImportFrom(module='tensorflow.keras.optimizers',names=[alias(name='get',asname='get_optimizer_by_name')],level=0)
    ]

    ## HOROVOD CONFIGS
    configs = [
        Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='init',ctx=Load()),args=[],keywords=[])),
        Assign(targets=[Name(id='config',ctx=Store())],value=Call(func=Attribute(value=Name(id='tf',ctx=Load()),attr='ConfigProto',ctx=Load()),args=[],keywords=[])),
        Assign(targets=[Attribute(value=Attribute(value=Name(id='config',ctx=Load()),attr='gpu_options',ctx=Load()),attr='allow_growth',ctx=Store())],value=NameConstant(value=True)),
        Assign(targets=[Attribute(value=Attribute(value=Name(id='config',ctx=Load()),attr='gpu_options',ctx=Load()),attr='visible_device_list',ctx=Store())],value=Call(func=Name(id='str',ctx=Load()),args=[Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='local_rank',ctx=Load()),args=[],keywords=[])],keywords=[])),
        Expr(value=Call(func=Attribute(value=Name(id='K',ctx=Load()),attr='set_session',ctx=Load()),args=[Call(func=Attribute(value=Name(id='tf',ctx=Load()),attr='Session',ctx=Load()),args=[],keywords=[keyword(arg='config',value=Name(id='config',ctx=Load()))])],keywords=[]))
    ]

    ## Keywords
    optimizer_arg=Call(func=Name(id='adapt_optimizer',ctx=Load()),args=[],keywords=[])
    optimizer_keyword=keyword(arg='optimizer',value=Call(func=Name(id='adapt_optimizer',ctx=Load()),args=[Name(id='opt',ctx=Load())],keywords=[]))
    callbacks_keyword=keyword(arg='callbacks',value=Call(func=Name(id='adapt_callbacks',ctx=Load()),args=[List(elts=[],ctx=Load()),NameConstant(value=True)],keywords=[]))
    epochs_keyword=keyword(arg='epochs',value=Call(func=Name(id='adapt_epochs',ctx=Load()),args=[Name(id='epochs',ctx=Load())],keywords=[]))
    verbose_keyword=keyword(arg='verbose',value=IfExp(test=Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),body=Num(n=1),orelse=Num(n=0)))
    steps_per_epoch_keyword=keyword(arg='steps_per_epoch',value=Call(func=Name(id='adapt_steps',ctx=Load()),args=[],keywords=[]))
    validation_steps_keyword=keyword(arg='validation_steps',value=Call(func=Name(id='adapt_steps',ctx=Load()),args=[],keywords=[]))

    ## hvd conditions
    if_rank_0=If(test=Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),body=[],orelse=[])

class Torch:

## Torchvision model names:
    possible_model_names_torchvision = [
    'resnet18',
    'alexnet',
    'vgg16',
    'squeezenet1_0',
    'densenet161',
    'inception_v3',
    'googlenet',
    'shufflenet_v2_x1_0',
    'mobilenet_v2',
    'resnext50_32x4d',
    'wide_resnet50_2',
    'mnasnet1_0'
    ]

## Torchvision optimizer names:
    possible_optim_names_torchvision = [
    	'Adadelta',
    	'Adagrad',
    	'Adam',
    	'Adamax',
    	'AdamW',
    	'ASGD',
    	'LBFGS',
    	'RMSprop',
    	'Rprop',
    	'SGD',
    	'SparseAdam'
    ]

## IMPORTS:
    imports=[
        ImportFrom(module='horovodizer_helper_Torch',names=[alias(name='*',asname=None)],level=0),
        Import(names=[alias(name='horovod.torch',asname='hvd')]),
        Import(names=[alias(name='torch',asname=None)]),
        Import(names=[alias(name='torch.utils.data.distributed',asname=None)])
    ]

## HOROVOD CONFIGS
    configs = [
        Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='init',ctx=Load()),args=[],keywords=[])),
        Expr(value=Call(func=Attribute(value=Attribute(value=Name(id='torch',ctx=Load()),attr='cuda',ctx=Load()),attr='set_device',ctx=Load()),args=[Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='local_rank',ctx=Load()),args=[],keywords=[])],keywords=[]))
    ]

## CODE ADAPTATIONS

    # adapt data loaders:
    data_sampler=Assign( targets=[Name(id='hvd_sampler_', ctx=Store())], value=Call( func=Attribute( value=Attribute( value=Attribute( value=Attribute( value=Name( id='torch', ctx=Load()), attr='utils', ctx=Load()), attr='data', ctx=Load()), attr='distributed', ctx=Load()), attr='DistributedSampler', ctx=Load()), args=[], keywords=[ keyword( arg='dataset', value=Name( id='dataset', ctx=Load())), keyword( arg='num_replicas', value=Call( func=Attribute( value=Name( id='hvd', ctx=Load()), attr='size', ctx=Load()), args=[], keywords=[])), keyword( arg='rank', value=Call( func=Attribute( value=Name( id='hvd', ctx=Load()), attr='rank', ctx=Load()), args=[], keywords=[]))]))
    data_sampler_keyword=keyword(arg='sampler',value=Name(id='hvd_sampler_',ctx=Load()))

    # send model to cuda device
    model_to_cuda=Expr(value=Call(func=Attribute(value=Name(id='model',ctx=Load()),attr='cuda',ctx=Load()),args=[],keywords=[]))
    # adapt optimizer
    adapt_opt=Expr(value=Call(func=Name(id='adapt_optimizer',ctx=Load()),args=[Name(id='optimizer',ctx=Load()), Name(id='model',ctx=Load())],keywords=[]))
    # adapt loss & accuracy
    adapt_loss=[
        AugAssign( target=Name( id='test_loss', ctx=Store()), op=Div(), value=Call( func=Name( id='len', ctx=Load()), args=[Name( id='test_sampler', ctx=Load())], keywords=[])),
        Assign( targets=[Name( id='test_loss', ctx=Store())], value=Call( func=Name( id='hvd_metric_average', ctx=Load()), args=[ Name( id='test_loss', ctx=Load()), Str(s='avg_loss')], keywords=[]))
    ]
    adapt_accuracy=[
        AugAssign( target=Name( id='test_accuracy', ctx=Store()), op=Div(), value=Call( func=Name( id='len', ctx=Load()), args=[Name( id='test_sampler', ctx=Load())], keywords=[])),
        Assign( targets=[Name( id='test_accuracy', ctx=Store())], value=Call( func=Name( id='hvd_metric_average', ctx=Load()), args=[ Name( id='test_accuracy', ctx=Load()), Str(s='avg_accuracy')], keywords=[]))
    ]

    ## hvd conditions
    if_rank_0=If(test=Compare(left=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='rank',ctx=Load()),args=[],keywords=[]),ops=[Eq()],comparators=[Num(n=0)]),body=[],orelse=[])

    broadcast_parameters=Expr(value=Call(func=Attribute(value=Name(id='hvd',ctx=Load()),attr='broadcast_parameters',ctx=Load()),args=[Call(func=Attribute(value=Name(id='model', ctx=Load()),attr='state_dict',ctx=Load()),args=[],keywords=[])],keywords=[keyword(arg='root_rank',value=Num(n=0))]))

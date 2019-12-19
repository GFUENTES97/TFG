Module(body=[
  Import(names=[alias(
    name='torch',
    asname=None)]),
  Import(names=[alias(
    name='torch.nn',
    asname='nn')]),
  Import(names=[alias(
    name='torchvision.transforms',
    asname='transforms')]),
  Import(names=[alias(
    name='torchvision.datasets',
    asname='dsets')]),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='torch',
        ctx=Load()),
      attr='manual_seed',
      ctx=Load()),
    args=[Num(n=0)],
    keywords=[])),
  ImportFrom(
    module='torch.optim.lr_scheduler',
    names=[alias(
      name='StepLR',
      asname=None)],
    level=0),
  Expr(value=Str(s='\nSTEP 1: LOADING DATASET\n')),
  Assign(
    targets=[Name(
      id='train_dataset',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='dsets',
          ctx=Load()),
        attr='MNIST',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='root',
          value=Str(s='./data')),
        keyword(
          arg='train',
          value=NameConstant(value=True)),
        keyword(
          arg='transform',
          value=Call(
            func=Attribute(
              value=Name(
                id='transforms',
                ctx=Load()),
              attr='ToTensor',
              ctx=Load()),
            args=[],
            keywords=[])),
        keyword(
          arg='download',
          value=NameConstant(value=True))])),
  Assign(
    targets=[Name(
      id='test_dataset',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='dsets',
          ctx=Load()),
        attr='MNIST',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='root',
          value=Str(s='./data')),
        keyword(
          arg='train',
          value=NameConstant(value=False)),
        keyword(
          arg='transform',
          value=Call(
            func=Attribute(
              value=Name(
                id='transforms',
                ctx=Load()),
              attr='ToTensor',
              ctx=Load()),
            args=[],
            keywords=[]))])),
  Expr(value=Str(s='\nSTEP 2: MAKING DATASET ITERABLE\n')),
  Assign(
    targets=[Name(
      id='batch_size',
      ctx=Store())],
    value=Num(n=100)),
  Assign(
    targets=[Name(
      id='n_iters',
      ctx=Store())],
    value=Num(n=3000)),
  Assign(
    targets=[Name(
      id='num_epochs',
      ctx=Store())],
    value=BinOp(
      left=Name(
        id='n_iters',
        ctx=Load()),
      op=Div(),
      right=BinOp(
        left=Call(
          func=Name(
            id='len',
            ctx=Load()),
          args=[Name(
            id='train_dataset',
            ctx=Load())],
          keywords=[]),
        op=Div(),
        right=Name(
          id='batch_size',
          ctx=Load())))),
  Assign(
    targets=[Name(
      id='num_epochs',
      ctx=Store())],
    value=Call(
      func=Name(
        id='int',
        ctx=Load()),
      args=[Name(
        id='num_epochs',
        ctx=Load())],
      keywords=[])),
  Assign(
    targets=[Name(
      id='train_loader',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Attribute(
            value=Name(
              id='torch',
              ctx=Load()),
            attr='utils',
            ctx=Load()),
          attr='data',
          ctx=Load()),
        attr='DataLoader',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='dataset',
          value=Name(
            id='train_dataset',
            ctx=Load())),
        keyword(
          arg='batch_size',
          value=Name(
            id='batch_size',
            ctx=Load())),
        keyword(
          arg='shuffle',
          value=NameConstant(value=True))])),
  Assign(
    targets=[Name(
      id='test_loader',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Attribute(
            value=Name(
              id='torch',
              ctx=Load()),
            attr='utils',
            ctx=Load()),
          attr='data',
          ctx=Load()),
        attr='DataLoader',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='dataset',
          value=Name(
            id='test_dataset',
            ctx=Load())),
        keyword(
          arg='batch_size',
          value=Name(
            id='batch_size',
            ctx=Load())),
        keyword(
          arg='shuffle',
          value=NameConstant(value=False))])),
  Expr(value=Str(s='\nSTEP 3: CREATE MODEL CLASS\n')),
  ClassDef(
    name='FeedforwardNeuralNetModel',
    bases=[Attribute(
      value=Name(
        id='nn',
        ctx=Load()),
      attr='Module',
      ctx=Load())],
    keywords=[],
    body=[
      FunctionDef(
        name='__init__',
        args=arguments(
          args=[
            arg(
              arg='self',
              annotation=None),
            arg(
              arg='input_dim',
              annotation=None),
            arg(
              arg='hidden_dim',
              annotation=None),
            arg(
              arg='output_dim',
              annotation=None)],
          vararg=None,
          kwonlyargs=[],
          kw_defaults=[],
          kwarg=None,
          defaults=[]),
        body=[
          Expr(value=Call(
            func=Attribute(
              value=Call(
                func=Name(
                  id='super',
                  ctx=Load()),
                args=[
                  Name(
                    id='FeedforwardNeuralNetModel',
                    ctx=Load()),
                  Name(
                    id='self',
                    ctx=Load())],
                keywords=[]),
              attr='__init__',
              ctx=Load()),
            args=[],
            keywords=[])),
          Assign(
            targets=[Attribute(
              value=Name(
                id='self',
                ctx=Load()),
              attr='fc1',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='nn',
                  ctx=Load()),
                attr='Linear',
                ctx=Load()),
              args=[
                Name(
                  id='input_dim',
                  ctx=Load()),
                Name(
                  id='hidden_dim',
                  ctx=Load())],
              keywords=[])),
          Assign(
            targets=[Attribute(
              value=Name(
                id='self',
                ctx=Load()),
              attr='relu',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='nn',
                  ctx=Load()),
                attr='ReLU',
                ctx=Load()),
              args=[],
              keywords=[])),
          Assign(
            targets=[Attribute(
              value=Name(
                id='self',
                ctx=Load()),
              attr='fc2',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='nn',
                  ctx=Load()),
                attr='Linear',
                ctx=Load()),
              args=[
                Name(
                  id='hidden_dim',
                  ctx=Load()),
                Name(
                  id='output_dim',
                  ctx=Load())],
              keywords=[]))],
        decorator_list=[],
        returns=None),
      FunctionDef(
        name='forward',
        args=arguments(
          args=[
            arg(
              arg='self',
              annotation=None),
            arg(
              arg='x',
              annotation=None)],
          vararg=None,
          kwonlyargs=[],
          kw_defaults=[],
          kwarg=None,
          defaults=[]),
        body=[
          Assign(
            targets=[Name(
              id='out',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='self',
                  ctx=Load()),
                attr='fc1',
                ctx=Load()),
              args=[Name(
                id='x',
                ctx=Load())],
              keywords=[])),
          Assign(
            targets=[Name(
              id='out',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='self',
                  ctx=Load()),
                attr='relu',
                ctx=Load()),
              args=[Name(
                id='out',
                ctx=Load())],
              keywords=[])),
          Assign(
            targets=[Name(
              id='out',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='self',
                  ctx=Load()),
                attr='fc2',
                ctx=Load()),
              args=[Name(
                id='out',
                ctx=Load())],
              keywords=[])),
          Return(value=Name(
            id='out',
            ctx=Load()))],
        decorator_list=[],
        returns=None)],
    decorator_list=[]),
  Expr(value=Str(s='\nSTEP 4: INSTANTIATE MODEL CLASS\n')),
  Assign(
    targets=[Name(
      id='input_dim',
      ctx=Store())],
    value=BinOp(
      left=Num(n=28),
      op=Mult(),
      right=Num(n=28))),
  Assign(
    targets=[Name(
      id='hidden_dim',
      ctx=Store())],
    value=Num(n=100)),
  Assign(
    targets=[Name(
      id='output_dim',
      ctx=Store())],
    value=Num(n=10)),
  Assign(
    targets=[Name(
      id='model',
      ctx=Store())],
    value=Call(
      func=Name(
        id='FeedforwardNeuralNetModel',
        ctx=Load()),
      args=[
        Name(
          id='input_dim',
          ctx=Load()),
        Name(
          id='hidden_dim',
          ctx=Load()),
        Name(
          id='output_dim',
          ctx=Load())],
      keywords=[])),
  Expr(value=Str(s='\nSTEP 5: INSTANTIATE LOSS CLASS\n')),
  Assign(
    targets=[Name(
      id='criterion',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='nn',
          ctx=Load()),
        attr='CrossEntropyLoss',
        ctx=Load()),
      args=[],
      keywords=[])),
  Expr(value=Str(s='\nSTEP 6: INSTANTIATE OPTIMIZER CLASS\n')),
  Assign(
    targets=[Name(
      id='learning_rate',
      ctx=Store())],
    value=Num(n=0.1)),
  Assign(
    targets=[Name(
      id='optimizer',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Name(
            id='torch',
            ctx=Load()),
          attr='optim',
          ctx=Load()),
        attr='SGD',
        ctx=Load()),
      args=[Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='parameters',
          ctx=Load()),
        args=[],
        keywords=[])],
      keywords=[
        keyword(
          arg='lr',
          value=Name(
            id='learning_rate',
            ctx=Load())),
        keyword(
          arg='momentum',
          value=Num(n=0.9)),
        keyword(
          arg='nesterov',
          value=NameConstant(value=True))])),
  Expr(value=Str(s='\nSTEP 7: INSTANTIATE STEP LEARNING SCHEDULER CLASS\n')),
  Assign(
    targets=[Name(
      id='scheduler',
      ctx=Store())],
    value=Call(
      func=Name(
        id='StepLR',
        ctx=Load()),
      args=[Name(
        id='optimizer',
        ctx=Load())],
      keywords=[
        keyword(
          arg='step_size',
          value=Num(n=1)),
        keyword(
          arg='gamma',
          value=Num(n=0.1))])),
  Expr(value=Str(s='\nSTEP 7: TRAIN THE MODEL\n')),
  Assign(
    targets=[Name(
      id='iter',
      ctx=Store())],
    value=Num(n=0)),
  For(
    target=Name(
      id='epoch',
      ctx=Store()),
    iter=Call(
      func=Name(
        id='range',
        ctx=Load()),
      args=[Name(
        id='num_epochs',
        ctx=Load())],
      keywords=[]),
    body=[
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='scheduler',
            ctx=Load()),
          attr='step',
          ctx=Load()),
        args=[],
        keywords=[])),
      Expr(value=Call(
        func=Name(
          id='print',
          ctx=Load()),
        args=[
          Str(s='Epoch:'),
          Name(
            id='epoch',
            ctx=Load()),
          Str(s='LR:'),
          Call(
            func=Attribute(
              value=Name(
                id='scheduler',
                ctx=Load()),
              attr='get_lr',
              ctx=Load()),
            args=[],
            keywords=[])],
        keywords=[])),
      For(
        target=Tuple(
          elts=[
            Name(
              id='i',
              ctx=Store()),
            Tuple(
              elts=[
                Name(
                  id='images',
                  ctx=Store()),
                Name(
                  id='labels',
                  ctx=Store())],
              ctx=Store())],
          ctx=Store()),
        iter=Call(
          func=Name(
            id='enumerate',
            ctx=Load()),
          args=[Name(
            id='train_loader',
            ctx=Load())],
          keywords=[]),
        body=[
          Assign(
            targets=[Name(
              id='images',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Call(
                  func=Attribute(
                    value=Name(
                      id='images',
                      ctx=Load()),
                    attr='view',
                    ctx=Load()),
                  args=[
                    UnaryOp(
                      op=USub(),
                      operand=Num(n=1)),
                    BinOp(
                      left=Num(n=28),
                      op=Mult(),
                      right=Num(n=28))],
                  keywords=[]),
                attr='requires_grad_',
                ctx=Load()),
              args=[],
              keywords=[])),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='optimizer',
                ctx=Load()),
              attr='zero_grad',
              ctx=Load()),
            args=[],
            keywords=[])),
          Assign(
            targets=[Name(
              id='outputs',
              ctx=Store())],
            value=Call(
              func=Name(
                id='model',
                ctx=Load()),
              args=[Name(
                id='images',
                ctx=Load())],
              keywords=[])),
          Assign(
            targets=[Name(
              id='loss',
              ctx=Store())],
            value=Call(
              func=Name(
                id='criterion',
                ctx=Load()),
              args=[
                Name(
                  id='outputs',
                  ctx=Load()),
                Name(
                  id='labels',
                  ctx=Load())],
              keywords=[])),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='loss',
                ctx=Load()),
              attr='backward',
              ctx=Load()),
            args=[],
            keywords=[])),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='optimizer',
                ctx=Load()),
              attr='step',
              ctx=Load()),
            args=[],
            keywords=[])),
          AugAssign(
            target=Name(
              id='iter',
              ctx=Store()),
            op=Add(),
            value=Num(n=1)),
          If(
            test=Compare(
              left=BinOp(
                left=Name(
                  id='iter',
                  ctx=Load()),
                op=Mod(),
                right=Num(n=500)),
              ops=[Eq()],
              comparators=[Num(n=0)]),
            body=[
              Assign(
                targets=[Name(
                  id='correct',
                  ctx=Store())],
                value=Num(n=0)),
              Assign(
                targets=[Name(
                  id='total',
                  ctx=Store())],
                value=Num(n=0)),
              For(
                target=Tuple(
                  elts=[
                    Name(
                      id='images',
                      ctx=Store()),
                    Name(
                      id='labels',
                      ctx=Store())],
                  ctx=Store()),
                iter=Name(
                  id='test_loader',
                  ctx=Load()),
                body=[
                  Assign(
                    targets=[Name(
                      id='images',
                      ctx=Store())],
                    value=Call(
                      func=Attribute(
                        value=Name(
                          id='images',
                          ctx=Load()),
                        attr='view',
                        ctx=Load()),
                      args=[
                        UnaryOp(
                          op=USub(),
                          operand=Num(n=1)),
                        BinOp(
                          left=Num(n=28),
                          op=Mult(),
                          right=Num(n=28))],
                      keywords=[])),
                  Assign(
                    targets=[Name(
                      id='outputs',
                      ctx=Store())],
                    value=Call(
                      func=Name(
                        id='model',
                        ctx=Load()),
                      args=[Name(
                        id='images',
                        ctx=Load())],
                      keywords=[])),
                  Assign(
                    targets=[Tuple(
                      elts=[
                        Name(
                          id='_',
                          ctx=Store()),
                        Name(
                          id='predicted',
                          ctx=Store())],
                      ctx=Store())],
                    value=Call(
                      func=Attribute(
                        value=Name(
                          id='torch',
                          ctx=Load()),
                        attr='max',
                        ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(
                            id='outputs',
                            ctx=Load()),
                          attr='data',
                          ctx=Load()),
                        Num(n=1)],
                      keywords=[])),
                  AugAssign(
                    target=Name(
                      id='total',
                      ctx=Store()),
                    op=Add(),
                    value=Call(
                      func=Attribute(
                        value=Name(
                          id='labels',
                          ctx=Load()),
                        attr='size',
                        ctx=Load()),
                      args=[Num(n=0)],
                      keywords=[])),
                  AugAssign(
                    target=Name(
                      id='correct',
                      ctx=Store()),
                    op=Add(),
                    value=Call(
                      func=Attribute(
                        value=Compare(
                          left=Name(
                            id='predicted',
                            ctx=Load()),
                          ops=[Eq()],
                          comparators=[Name(
                            id='labels',
                            ctx=Load())]),
                        attr='sum',
                        ctx=Load()),
                      args=[],
                      keywords=[]))],
                orelse=[]),
              Assign(
                targets=[Name(
                  id='accuracy',
                  ctx=Store())],
                value=BinOp(
                  left=BinOp(
                    left=Num(n=100),
                    op=Mult(),
                    right=Name(
                      id='correct',
                      ctx=Load())),
                  op=Div(),
                  right=Name(
                    id='total',
                    ctx=Load()))),
              Expr(value=Call(
                func=Name(
                  id='print',
                  ctx=Load()),
                args=[Call(
                  func=Attribute(
                    value=Str(s='Iteration: {}. Loss: {}. Accuracy: {}'),
                    attr='format',
                    ctx=Load()),
                  args=[
                    Name(
                      id='iter',
                      ctx=Load()),
                    Call(
                      func=Attribute(
                        value=Name(
                          id='loss',
                          ctx=Load()),
                        attr='item',
                        ctx=Load()),
                      args=[],
                      keywords=[]),
                    Name(
                      id='accuracy',
                      ctx=Load())],
                  keywords=[])],
                keywords=[]))],
            orelse=[])],
        orelse=[])],
    orelse=[])])
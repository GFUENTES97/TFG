Module(body=[
  ImportFrom(
    module='engine',
    names=[
      alias(
        name='train_one_epoch',
        asname=None),
      alias(
        name='evaluate',
        asname=None)],
    level=0),
  Import(names=[alias(
    name='utils',
    asname=None)]),
  FunctionDef(
    name='main',
    args=arguments(
      args=[],
      vararg=None,
      kwonlyargs=[],
      kw_defaults=[],
      kwarg=None,
      defaults=[]),
    body=[
      Assign(
        targets=[Name(
          id='device',
          ctx=Store())],
        value=IfExp(
          test=Call(
            func=Attribute(
              value=Attribute(
                value=Name(
                  id='torch',
                  ctx=Load()),
                attr='cuda',
                ctx=Load()),
              attr='is_available',
              ctx=Load()),
            args=[],
            keywords=[]),
          body=Call(
            func=Attribute(
              value=Name(
                id='torch',
                ctx=Load()),
              attr='device',
              ctx=Load()),
            args=[Str(s='cuda')],
            keywords=[]),
          orelse=Call(
            func=Attribute(
              value=Name(
                id='torch',
                ctx=Load()),
              attr='device',
              ctx=Load()),
            args=[Str(s='cpu')],
            keywords=[]))),
      Assign(
        targets=[Name(
          id='num_classes',
          ctx=Store())],
        value=Num(n=2)),
      Assign(
        targets=[Name(
          id='dataset',
          ctx=Store())],
        value=Call(
          func=Name(
            id='PennFudanDataset',
            ctx=Load()),
          args=[
            Str(s='PennFudanPed'),
            Call(
              func=Name(
                id='get_transform',
                ctx=Load()),
              args=[],
              keywords=[keyword(
                arg='train',
                value=NameConstant(value=True))])],
          keywords=[])),
      Assign(
        targets=[Name(
          id='dataset_test',
          ctx=Store())],
        value=Call(
          func=Name(
            id='PennFudanDataset',
            ctx=Load()),
          args=[
            Str(s='PennFudanPed'),
            Call(
              func=Name(
                id='get_transform',
                ctx=Load()),
              args=[],
              keywords=[keyword(
                arg='train',
                value=NameConstant(value=False))])],
          keywords=[])),
      Assign(
        targets=[Name(
          id='indices',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Call(
              func=Attribute(
                value=Name(
                  id='torch',
                  ctx=Load()),
                attr='randperm',
                ctx=Load()),
              args=[Call(
                func=Name(
                  id='len',
                  ctx=Load()),
                args=[Name(
                  id='dataset',
                  ctx=Load())],
                keywords=[])],
              keywords=[]),
            attr='tolist',
            ctx=Load()),
          args=[],
          keywords=[])),
      Assign(
        targets=[Name(
          id='dataset',
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
            attr='Subset',
            ctx=Load()),
          args=[
            Name(
              id='dataset',
              ctx=Load()),
            Subscript(
              value=Name(
                id='indices',
                ctx=Load()),
              slice=Slice(
                lower=None,
                upper=UnaryOp(
                  op=USub(),
                  operand=Num(n=50)),
                step=None),
              ctx=Load())],
          keywords=[])),
      Assign(
        targets=[Name(
          id='dataset_test',
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
            attr='Subset',
            ctx=Load()),
          args=[
            Name(
              id='dataset_test',
              ctx=Load()),
            Subscript(
              value=Name(
                id='indices',
                ctx=Load()),
              slice=Slice(
                lower=UnaryOp(
                  op=USub(),
                  operand=Num(n=50)),
                upper=None,
                step=None),
              ctx=Load())],
          keywords=[])),
      Assign(
        targets=[Name(
          id='data_loader',
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
          args=[Name(
            id='dataset',
            ctx=Load())],
          keywords=[
            keyword(
              arg='batch_size',
              value=Num(n=2)),
            keyword(
              arg='shuffle',
              value=NameConstant(value=True)),
            keyword(
              arg='num_workers',
              value=Num(n=4)),
            keyword(
              arg='collate_fn',
              value=Attribute(
                value=Name(
                  id='utils',
                  ctx=Load()),
                attr='collate_fn',
                ctx=Load()))])),
      Assign(
        targets=[Name(
          id='data_loader_test',
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
          args=[Name(
            id='dataset_test',
            ctx=Load())],
          keywords=[
            keyword(
              arg='batch_size',
              value=Num(n=1)),
            keyword(
              arg='shuffle',
              value=NameConstant(value=False)),
            keyword(
              arg='num_workers',
              value=Num(n=4)),
            keyword(
              arg='collate_fn',
              value=Attribute(
                value=Name(
                  id='utils',
                  ctx=Load()),
                attr='collate_fn',
                ctx=Load()))])),
      Assign(
        targets=[Name(
          id='model',
          ctx=Store())],
        value=Call(
          func=Name(
            id='get_model_instance_segmentation',
            ctx=Load()),
          args=[Name(
            id='num_classes',
            ctx=Load())],
          keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='to',
          ctx=Load()),
        args=[Name(
          id='device',
          ctx=Load())],
        keywords=[])),
      Assign(
        targets=[Name(
          id='params',
          ctx=Store())],
        value=ListComp(
          elt=Name(
            id='p',
            ctx=Load()),
          generators=[comprehension(
            target=Name(
              id='p',
              ctx=Store()),
            iter=Call(
              func=Attribute(
                value=Name(
                  id='model',
                  ctx=Load()),
                attr='parameters',
                ctx=Load()),
              args=[],
              keywords=[]),
            ifs=[Attribute(
              value=Name(
                id='p',
                ctx=Load()),
              attr='requires_grad',
              ctx=Load())],
            is_async=0)])),
      Assign(
        targets=[Name(
          id='opt',
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
          args=[Name(
            id='params',
            ctx=Load())],
          keywords=[
            keyword(
              arg='lr',
              value=Num(n=0.005)),
            keyword(
              arg='momentum',
              value=Num(n=0.9)),
            keyword(
              arg='weight_decay',
              value=Num(n=0.0005))])),
      Assign(
        targets=[Name(
          id='lr_scheduler',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Attribute(
              value=Attribute(
                value=Name(
                  id='torch',
                  ctx=Load()),
                attr='optim',
                ctx=Load()),
              attr='lr_scheduler',
              ctx=Load()),
            attr='StepLR',
            ctx=Load()),
          args=[Name(
            id='opt',
            ctx=Load())],
          keywords=[
            keyword(
              arg='step_size',
              value=Num(n=3)),
            keyword(
              arg='gamma',
              value=Num(n=0.1))])),
      Assign(
        targets=[Name(
          id='num_epochs',
          ctx=Store())],
        value=Num(n=10)),
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
            func=Name(
              id='train_one_epoch',
              ctx=Load()),
            args=[
              Name(
                id='model',
                ctx=Load()),
              Name(
                id='opt',
                ctx=Load()),
              Name(
                id='data_loader',
                ctx=Load()),
              Name(
                id='device',
                ctx=Load()),
              Name(
                id='epoch',
                ctx=Load())],
            keywords=[keyword(
              arg='print_freq',
              value=Num(n=10))])),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='lr_scheduler',
                ctx=Load()),
              attr='step',
              ctx=Load()),
            args=[],
            keywords=[])),
          Expr(value=Call(
            func=Name(
              id='evaluate',
              ctx=Load()),
            args=[
              Name(
                id='model',
                ctx=Load()),
              Name(
                id='data_loader_test',
                ctx=Load())],
            keywords=[keyword(
              arg='device',
              value=Name(
                id='device',
                ctx=Load()))]))],
        orelse=[]),
      Expr(value=Call(
        func=Name(
          id='print',
          ctx=Load()),
        args=[Str(s="That's it!")],
        keywords=[]))],
    decorator_list=[],
    returns=None)])
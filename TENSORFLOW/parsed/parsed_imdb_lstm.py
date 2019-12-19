Module(body=[
  Expr(value=Str(s="\n#Trains an LSTM model on the IMDB sentiment classification task.\n\nThe dataset is actually too small for LSTM to be of any advantage\ncompared to simpler, much faster methods such as TF-IDF + LogReg.\n\n**Notes**\n\n- RNNs are tricky. Choice of batch size is important,\nchoice of loss and optimizer is critical, etc.\nSome configurations won't converge.\n\n- LSTM loss decrease patterns during training can be quite different\nfrom what you see with CNNs/MLPs/etc.\n\n")),
  ImportFrom(
    module='__future__',
    names=[alias(
      name='print_function',
      asname=None)],
    level=0),
  ImportFrom(
    module='keras.preprocessing',
    names=[alias(
      name='sequence',
      asname=None)],
    level=0),
  ImportFrom(
    module='keras.models',
    names=[alias(
      name='Sequential',
      asname=None)],
    level=0),
  ImportFrom(
    module='keras.layers',
    names=[
      alias(
        name='Dense',
        asname=None),
      alias(
        name='Embedding',
        asname=None)],
    level=0),
  ImportFrom(
    module='keras.layers',
    names=[alias(
      name='LSTM',
      asname=None)],
    level=0),
  ImportFrom(
    module='keras.datasets',
    names=[alias(
      name='imdb',
      asname=None)],
    level=0),
  Assign(
    targets=[Name(
      id='max_features',
      ctx=Store())],
    value=Num(n=20000)),
  Assign(
    targets=[Name(
      id='maxlen',
      ctx=Store())],
    value=Num(n=80)),
  Assign(
    targets=[Name(
      id='batch_size',
      ctx=Store())],
    value=Num(n=32)),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[Str(s='Loading data...')],
    keywords=[])),
  Assign(
    targets=[Tuple(
      elts=[
        Tuple(
          elts=[
            Name(
              id='x_train',
              ctx=Store()),
            Name(
              id='y_train',
              ctx=Store())],
          ctx=Store()),
        Tuple(
          elts=[
            Name(
              id='x_test',
              ctx=Store()),
            Name(
              id='y_test',
              ctx=Store())],
          ctx=Store())],
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='imdb',
          ctx=Load()),
        attr='load_data',
        ctx=Load()),
      args=[],
      keywords=[keyword(
        arg='num_words',
        value=Name(
          id='max_features',
          ctx=Load()))])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Call(
        func=Name(
          id='len',
          ctx=Load()),
        args=[Name(
          id='x_train',
          ctx=Load())],
        keywords=[]),
      Str(s='train sequences')],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Call(
        func=Name(
          id='len',
          ctx=Load()),
        args=[Name(
          id='x_test',
          ctx=Load())],
        keywords=[]),
      Str(s='test sequences')],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[Str(s='Pad sequences (samples x time)')],
    keywords=[])),
  Assign(
    targets=[Name(
      id='x_train',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='sequence',
          ctx=Load()),
        attr='pad_sequences',
        ctx=Load()),
      args=[Name(
        id='x_train',
        ctx=Load())],
      keywords=[keyword(
        arg='maxlen',
        value=Name(
          id='maxlen',
          ctx=Load()))])),
  Assign(
    targets=[Name(
      id='x_test',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='sequence',
          ctx=Load()),
        attr='pad_sequences',
        ctx=Load()),
      args=[Name(
        id='x_test',
        ctx=Load())],
      keywords=[keyword(
        arg='maxlen',
        value=Name(
          id='maxlen',
          ctx=Load()))])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='x_train shape:'),
      Attribute(
        value=Name(
          id='x_train',
          ctx=Load()),
        attr='shape',
        ctx=Load())],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='x_test shape:'),
      Attribute(
        value=Name(
          id='x_test',
          ctx=Load()),
        attr='shape',
        ctx=Load())],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[Str(s='Build model...')],
    keywords=[])),
  Assign(
    targets=[Name(
      id='model',
      ctx=Store())],
    value=Call(
      func=Name(
        id='Sequential',
        ctx=Load()),
      args=[],
      keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='model',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='Embedding',
        ctx=Load()),
      args=[
        Name(
          id='max_features',
          ctx=Load()),
        Num(n=128)],
      keywords=[])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='model',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='LSTM',
        ctx=Load()),
      args=[Num(n=128)],
      keywords=[
        keyword(
          arg='dropout',
          value=Num(n=0.2)),
        keyword(
          arg='recurrent_dropout',
          value=Num(n=0.2))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='model',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='Dense',
        ctx=Load()),
      args=[Num(n=1)],
      keywords=[keyword(
        arg='activation',
        value=Str(s='sigmoid'))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='model',
        ctx=Load()),
      attr='compile',
      ctx=Load()),
    args=[],
    keywords=[
      keyword(
        arg='loss',
        value=Str(s='binary_crossentropy')),
      keyword(
        arg='optimizer',
        value=Str(s='adam')),
      keyword(
        arg='metrics',
        value=List(
          elts=[Str(s='accuracy')],
          ctx=Load()))])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[Str(s='Train...')],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='model',
        ctx=Load()),
      attr='fit',
      ctx=Load()),
    args=[
      Name(
        id='x_train',
        ctx=Load()),
      Name(
        id='y_train',
        ctx=Load())],
    keywords=[
      keyword(
        arg='batch_size',
        value=Name(
          id='batch_size',
          ctx=Load())),
      keyword(
        arg='epochs',
        value=Num(n=15)),
      keyword(
        arg='validation_data',
        value=Tuple(
          elts=[
            Name(
              id='x_test',
              ctx=Load()),
            Name(
              id='y_test',
              ctx=Load())],
          ctx=Load()))])),
  Assign(
    targets=[Tuple(
      elts=[
        Name(
          id='score',
          ctx=Store()),
        Name(
          id='acc',
          ctx=Store())],
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='model',
          ctx=Load()),
        attr='evaluate',
        ctx=Load()),
      args=[
        Name(
          id='x_test',
          ctx=Load()),
        Name(
          id='y_test',
          ctx=Load())],
      keywords=[keyword(
        arg='batch_size',
        value=Name(
          id='batch_size',
          ctx=Load()))])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='Test score:'),
      Name(
        id='score',
        ctx=Load())],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='Test accuracy:'),
      Name(
        id='acc',
        ctx=Load())],
    keywords=[]))])
class Hyperparameters:
    def __init__(self):
        self.model = 'pointnet2_cls_ssg'  # Model to use [pointnet2_cls_ssg, pointnet2_cls_msg]
        self.batch_size = 64  # Size of batch
        self.nepochs = 300  # Number of epochs to train for
        self.nclasses = 40  # Number of classes in the dataset
        self.lr = 0.005  # Learning rate
        self.decay_rate = 5e-4  # Rate of decay for learning rate
        self.step_size = 10  # Step size for learning rate decay
        self.gamma = 0.9  # Gamma for learning rate decay
        self.log_dir = 'logs'  # Log directory
        self.checkpoint_interval = 10  # Interval to save model checkpoints
        self.optimizer = 'adam'  # Optimizer to use
        self.beta1 = 0.9  # Momentum-like term for Adam optimizer
        self.beta2 = 0.999  # Adaptive learning rate term for Adam optimizer
        self.dropout_rate1 = 0.5  # First dropout rate , not used see model.pointnet 
        self.dropout_rate2 = 0.5  # Second dropout rate , not used see model.pointet
        self.npoints = 1024
     def to_dict(self):
        
        return vars(self)

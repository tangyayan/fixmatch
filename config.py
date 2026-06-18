import torch

class Config():
    def __init__(self):
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # model
        self.model_name = 'wideresnet'
        self.depth = 28
        self.widen_factor = 2
        self.dropout_rate = 0.0

        # train
        self.num_steps = 2000
        self.batch_size = 64
        self.learning_rate = 0.03
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.optimizer = 'sgd'
        self.schedule = {
            'scheduler': 'cosine',
            # 'step_size': 30, 'gamma': 0.1,
        }

        # loss
        self.lambda_u = 1.0
        self.tao = 0.95
        self.T = 1

        # dataset
        self.num_classes = 10
        self.num_labels = 40
        self.mu = 7
        self.augment_type = 'randaugment'

        # print
        self.print_step = 100
        model_output_name = f'labels_{self.num_labels}_{self.model_name}_depth_{self.depth}_widen_{self.widen_factor}'
        self.save_dir = f'./result/{model_output_name}'
        self.checkpoint_dir = f'./checkpoint/{model_output_name}'
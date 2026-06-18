import semilearn
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer

def main(config:dict):
    config = get_config(config)
    # create model
    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

    # dataset
    dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir, include_lb_to_ulb=config.include_lb_to_ulb)
    train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
    train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
    eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size)

    # train
    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)

    # evaluate
    trainer.evaluate(eval_loader)

    # predict
    y_pred, y_logits = trainer.predict(eval_loader)

if __name__ == '__main__':
    num_labels = 40
    config = {
        'seed': 42,
        'algorithm': 'fixmatch',
        'net': 'wrn_28_2',
        'use_pretrain': False,
        # 'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',

        # optimization configs
        'epoch': 1024,
        'num_train_iter': 32768,
        'num_eval_iter': 5120,
        'num_log_iter': 256,
        'optim': 'SGD',
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'layer_decay': 1.0,
        'batch_size': 64,
        'eval_batch_size': 256,
        'clip': 0.0,


        # dataset configs
        'dataset': 'cifar10',
        'num_labels': num_labels,
        'num_classes': 10,
        'img_size': 32,
        'crop_ratio': 0.875,
        'data_dir': './dataset',
        'ulb_samples_per_class': None,

        # algorithm specific configs
        'hard_label': True,
        'T': 0.5,
        'p_cutoff': 0.95,
        'uratio': 7,
        'ulb_loss_ratio': 1.0,
        'ema_m': 0.999,
        'train_sampler': 'RandomSampler',

        # device configs
        'gpu': 0,
        'world_size': 1,
        'distributed': False,
        "num_workers": 2,

        # output
        'save_dir': './saved_models',
        'save_name': f'fixmatch_{num_labels}',
    }
    main(config)
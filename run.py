from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.itembasedcf import ItemBasedCF

if __name__ == '__main__':
    config = Config(
        model=ItemBasedCF,
        dataset='goodreads',
        config_file_list=["./config_itembasedcf.yaml"]
    )
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # 1) Full dataset
    full_dataset = create_dataset(config)
    logger.info(full_dataset)

    # 2) Split
    train_data, valid_data, test_data = data_preparation(config, full_dataset)

    # 3) Construct your model with BOTH
    model = ItemBasedCF(config, full_dataset, train_data).to(config['device'])
    logger.info(model)

    # 4) Trainer (with 0 epoch training for item_based_cf because there are no trainable params)
    trainer = Trainer(config, model)

    # 5) Fit + Evaluate
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

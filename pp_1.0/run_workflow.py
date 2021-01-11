from ecnet import Server
from ecnet.workflows.workflow_utils import find_optimal_num_inputs,\
    prop_range_from_split
from ecnet.utils.logging import logger

_NUM_PROC = 16


def main(db_name: str):

    # Set up logging
    logger.stream_level = 'info'
    logger.log_dir = db_name.replace('.csv', '') + '_logs'
    logger.file_level = 'debug'

    # Split database proportionally based on property value
    # Proportions are 70% learn, 20% validate, 10% test
    prop_range_from_split(db_name, [0.7, 0.2, 0.1])

    # Find the optimal number of input variables
    # Train (learn + valid) set used for evaluation
    n_desc = len(find_optimal_num_inputs(db_name, 'train', _NUM_PROC)[1])
    logger.log('info', 'Optimal number of input variables: {}'.format(n_desc))

    # Create server object with base config
    sv = Server(model_config=db_name.replace('.csv', '.yml'),
                num_processes=_NUM_PROC)

    # Load data
    sv.load_data(db_name)

    # Limit input variables to `n_desc` using Train set
    # Outputs to relevant database name
    sv.limit_inputs(
        n_desc, eval_set='train',
        output_filename=db_name.replace('.csv', '.{}.csv'.format(n_desc))
    )

    # Tune hyperparameters (architecture and ADAM)
    # 20 employer bees, 10 search cycles
    # Evaluation of solutions based on validation set median absolute error
    sv.tune_hyperparameters(20, 10, eval_set='valid', eval_fn='med_abs_error')

    # Create an ECNet project (saved and recalled later)
    # 5 pools with 75 trials/pool, best ANNs selected from each pool
    sv.create_project(db_name.replace('.csv', ''), 5, 75)

    # Train project
    # Select best candidates based on validation set median absolute error
    sv.train(validate=True, selection_set='valid',
             selection_fn='med_abs_error')

    # Obtain learning, validation, testing set median absolute error, r-squared
    err_l = sv.errors('med_abs_error', 'r2', dset='learn')
    err_v = sv.errors('med_abs_error', 'r2', dset='valid')
    err_t = sv.errors('med_abs_error', 'r2', dset='test')
    logger.log('info', 'Learning set performance: {}'.format(err_l))
    logger.log('info', 'Validation set performance: {}'.format(err_v))
    logger.log('info', 'Testing set performance: {}'.format(err_t))

    # Save the project, creating a .prj file and removing un-chosen candidates
    sv.save_project(del_candidates=True)

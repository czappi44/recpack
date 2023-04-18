import argparse
import os
import pandas as pd
import numpy as np
import recpack.pipelines
from recpack.matrix.interaction_matrix import InteractionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_dev_id', type=str, default='0')
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--test_path", type=str, default="")
parser.add_argument('--m', '--measure', type=int, nargs='+', default=[20])

parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--loss', default='cross-entropy', type=str)
parser.add_argument('--optim', default='adagrad', type=str)
parser.add_argument('--layers', default=1, type=int)
parser.add_argument('--embedding', default=100, type=int)
parser.add_argument('--hidden_size', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument("--dropout_p_embed", type=float, default=0.0)
parser.add_argument("--dropout_p_hidden", type=float, default=0.0)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--n_sample", type=int, default=2048)

parser.add_argument('--use_correct_weight_init', default=False, action='store_true')
parser.add_argument("--final_act", type=str, default="linear")
parser.add_argument("--bpreg", type=float, default=1.0)

args = parser.parse_args()

if __name__ == '__main__':
    item_id = 'ItemId'
    session_id = 'SessionId'
    time = 'Time'
    item_idx = 'ItemIdx'
    session_idx = 'SessionIdx'
    print(pd.DataFrame({'Args':list(args.__dict__.keys()), 'Values':list(args.__dict__.values())}))

    print(f"Loading train data from: {args.train_path}")
    train_df = pd.read_csv(args.train_path, sep='\t', dtype={item_id: np.int64, session_id: np.int64, time: np.int64})
    itemids = train_df[item_id].unique()
    itemidxs = pd.DataFrame({item_id: itemids, item_idx: np.arange(1, len(itemids)+1)})

    train_df = pd.merge(train_df, itemidxs, on=item_id)
    train_df = train_df[[session_id, item_idx, time]]
    train_df = train_df.sort_values([session_id, time, item_idx]).reset_index(drop=True)

    print(f"Loading test data from: {args.test_path}")
    test_df = pd.read_csv(args.test_path, sep='\t', dtype={item_id: np.int64, session_id: np.int64, time: np.int64})
    test_df = pd.merge(test_df, itemidxs, on=item_id)
    test_df = test_df[[session_id, item_idx, time]]
    test_df = test_df.sort_values([session_id, time, item_idx]).reset_index(drop=True)

    train_im = InteractionMatrix(train_df, user_ix=session_id, item_ix=item_idx, timestamp_ix=time)
    test_im = InteractionMatrix(test_df, user_ix=session_id, item_ix=item_idx, timestamp_ix=time)

    pipeline_builder = recpack.pipelines.PipelineBuilder('temp')
    pipeline_builder.set_validation_training_data(train_im) #comment in original code: "TODO In theory, if early stopping is not used, this could be the full training dataset."
    pipeline_builder.set_full_training_data(train_im) # this will not be used, but it must be set
    pipeline_builder.set_test_data([test_im, test_im])
    if args.loss == "cross-entropy":
        pipeline_builder.add_algorithm('GRU4RecCrossEntropy', params={'keep_last':True, 'max_epochs':args.n_epochs, 'optimization_algorithm':args.optim, 'num_layers':args.layers, 'num_components':args.embedding, 'hidden_size':args.hidden_size, 'batch_size':args.batch_size, 'dropout_p_embed':args.dropout_p_embed, 'dropout_p_hidden':args.dropout_p_hidden,'learning_rate':args.learning_rate, 'use_correct_weight_init':args.use_correct_weight_init, 'clipnorm': None})
    elif args.loss == "bpr-max":
        pipeline_builder.add_algorithm('GRU4RecNegSampling', params={'keep_last':True, 'max_epochs':args.n_epochs, 'loss_fn':args.loss, 'optimization_algorithm':args.optim, 'num_layers':args.layers, 'num_components':args.embedding, 'hidden_size':args.hidden_size, 'final_activation':args.final_act, 'batch_size':args.batch_size, 'dropout_p_embed':args.dropout_p_embed, 'dropout_p_hidden':args.dropout_p_hidden, 'learning_rate':args.learning_rate, 'num_negatives':args.n_sample, 'bpreg':args.bpreg, 'use_correct_weight_init':args.use_correct_weight_init, 'clipnorm': None})
    else:
        raise NotImplementedError
    pipeline_builder.add_metric('HitK', 10) 
    pipeline = pipeline_builder.build()
    pipeline.run(m=args.m)

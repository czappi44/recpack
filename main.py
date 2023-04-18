import argparse
import os
import pandas as pd
import numpy as np

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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_dev_id

import recpack.pipelines
from recpack.matrix.interaction_matrix import InteractionMatrix

if __name__ == '__main__':
    item_id = 'ItemId'
    session_id = 'SessionId'
    time = 'Time'
    item_idx = 'ItemIdx'
    session_idx = 'SessionIdx'
    print(pd.DataFrame({'Args':list(args.__dict__.keys()), 'Values':list(args.__dict__.values())}))

    print(f"Loading train data from: {args.train_path}")
    train_df = pd.read_csv(args.train_path, sep='\t', dtype={item_id: np.int64, session_id: np.int64, time: np.int64})
    # train_df = train_df.iloc[:10000]
    itemids = train_df[item_id].unique()
    itemidxs = pd.DataFrame({item_id: itemids, item_idx: np.arange(1, len(itemids)+1)})

    # sessids_train = np.sort(train_df[session_id].unique())
    # sessidxs_train = pd.DataFrame({session_id: sessids_train, session_idx: np.arange(1, len(sessids_train)+1)})

    train_df = pd.merge(train_df, itemidxs, on=item_id)
    # train_df = pd.merge(train_df, sessidxs_train, on=session_id)
    train_df = train_df[[session_id, item_idx, time]]
    train_df = train_df.sort_values([session_id, time, item_idx]).reset_index(drop=True)

    print(f"Loading test data from: {args.test_path}")
    test_df = pd.read_csv(args.test_path, sep='\t', dtype={item_id: np.int64, session_id: np.int64, time: np.int64})
    # test_df = test_df.iloc[:300]
    test_df = pd.merge(test_df, itemidxs, on=item_id)
    # test_df = pd.merge(test_df, sessidxs, on=session_id)
    test_df = test_df[[session_id, item_idx, time]]
    test_df = test_df.sort_values([session_id, time, item_idx]).reset_index(drop=True)
    # ####
    # print(train_df.head())
    # print(train_df[session_id].nunique())
    # print(train_df[item_idx].nunique())
    # print(test_df.head())
    # print(test_df[item_idx].nunique())
    # print(test_df[session_id].nunique())
    # ####

    train_im = InteractionMatrix(train_df, user_ix=session_id, item_ix=item_idx, timestamp_ix=time)
    test_im = InteractionMatrix(test_df, user_ix=session_id, item_ix=item_idx, timestamp_ix=time)

    print(f"\nTRAIN: shape: {train_im.shape}, max_sessionid: {train_im._df['uid'].max()}, max_itemidx: {train_im._df['iid'].max()}, nunique_sessionid: {train_im._df['uid'].nunique()}, nunique_itemidx: {train_im._df['iid'].nunique()}")
    print(f"TEST: shape: {test_im.shape}, max_sessionid: {test_im._df['uid'].max()}, max_itemidx: {test_im._df['iid'].max()}, nunique_sessionid: {test_im._df['uid'].nunique()}, nunique_itemidx: {test_im._df['iid'].nunique()}\n")
    # TODO: TALÁN itt tok mindegy hogy rosszul írja a sessionok / usereke számát TALÁN a gru4recnél nem használják semmrie, mert az alapján csak groupby van és annak tök mindegy

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
    print("Building pipeline...")
    pipeline = pipeline_builder.build()
    print("Running pipeline...")
    pipeline.run(m=args.m)

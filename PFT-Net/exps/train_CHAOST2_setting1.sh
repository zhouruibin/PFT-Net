#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=0
LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=$GPUID1
export CUDA_LAUNCH_BLOCKING=$LAUNCH_BLOCKING

###### Shared configs ######
DATASET='CHAOST2'
NWORKER=0
RUNS=1
ALL_EV=(1) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[1,2,3,4]
EXCLUDE_LABEL=None
USE_GT=False
###### Training configs ######
NSTEP=35000
DECAY=0.98
MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=5000 # interval for saving snapshot
# ===== if use valldation=====
USE_VAL=True
VAL_INTERVAL=5000
N_PART=3
SUPP_IDX=(2)
SEED=2021

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="train_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./exps_on_${DATASET}"
  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi
  python train_PFT.py with \
  mode='train' \
  dataset=$DATASET \
  num_workers=$NWORKER \
  n_steps=$NSTEP \
  eval_fold=$EVAL_FOLD \
  test_label=$TEST_LABEL \
  exclude_label=$EXCLUDE_LABEL \
  use_gt=$USE_GT \
  max_iters_per_load=$MAX_ITER \
  seed=$SEED \
  save_snapshot_every=$SNAPSHOT_INTERVAL \
  path.log_dir=$LOGDIR \
  use_validation=$USE_VAL \
  val_every=$VAL_INTERVAL \
  supp_idx=$SUPP_IDX \
  n_part=$N_PART \
  lr_step_gamma=$DECAY
done
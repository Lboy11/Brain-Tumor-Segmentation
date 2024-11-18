nnode=2
python \
    test1.py \
    -m \
    hydra.job.chdir=False \
    world_size=${nnode} \
    distributed=False \
    mode="train" \
    epochs=600 \
    eval_interval=1 \
    optim=adam \
    optim.lr=8e-4 \
    batch_size=2 \
    test_batch_size=2 \
    model="ensemble" \
    model.output="list" \
    model.feature="False" \
    model.width_ratio="0.5" \
    dataset="brats3d_acn" \
    loss="enumeration" \
    loss.missing_num="2" \
    loss.output="list" \
    workers=1 \
    gpu_ids="'0,1'"\
    trainer="trainer" \
    trainer.method="gmd" \
    checkname="gmd-enum-2-adam-acn"

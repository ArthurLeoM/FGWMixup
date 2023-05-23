
CUDA_VISIBLE_DEVICES=0  python -u ./src/gmixup_dgl.py --data_path . --backbone GCN --dataset NCI1 \
        --lr=1e-3 --gmixup True --seed=3407  --num_layers=6 --log_screen True --batch_size 128 --num_hidden 64 \
        --aug_ratio 0.25 --metric adj --agg=sum --pooling=mean --gpu --measure=uniform --kfold --symmetry --epoch=400 --beta_k=0.2 --alpha=0.95 --act=relu --bapg --rho=0.1 &


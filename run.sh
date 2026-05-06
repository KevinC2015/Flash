nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Flash \
    --category=Beauty \
    --n_codebook=32 \
    --lr=0.01 \
    --align_weight=0.1' \
    > logs_beauty.log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py \
    --model=Flash \
    --category=Toys_and_Games \
    --n_codebook=64 \
    --lr=0.003 \
    --align_weight=0.2' \
    > logs_Toys_and_Games.log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py \
    --model=Flash \
    --category=Sports_and_Outdoors \
    --n_codebook=64 \
    --n_embd=896 \
    --lr=0.003 \
    --align_weight=0.2' \
    > logs_Sports_and_Outdoors.log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py \
    --model=Flash \
    --category=CDs_and_Vinyl \
    --n_codebook=64 \
    --lr=0.001 \
    --codebook_size=256 \
    --align_weight=0.05' \
    > logs_CDs_and_Vinyl.log 2>&1 &

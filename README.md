
# Wait-k with Context-Consistency-Bi Training

# Clone our Project
```bash
git clone https://github.com/SAILAKKSHMEDOKKU/Wait-k---Context-consistency-model-SIMT.git
```

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**Installing Fairseq**
```bash
cd Waitk-Consistency-Bi-Training
pip install -r requirements.txt
pip install --editable .
```

# Training Pervasive Attention for IWSLT'14 De-En:
# Download and pre-process the dataset:

```bash
# Download and prepare the data
%cd Wait-k---Context-consistency-model-SIMT/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT="examples/translation/iwslt14_deen_bpe10k"
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

# Train Pervasive Attention on the pre-processed data:
```bash
MODEL="pa_iwslt_de_en"
mkdir -p checkpoints/$MODEL
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en -s de -t en \
    --user-dir examples/pervasive --arch pervasive \
    --max-source-positions 100  --max-target-positions 100 \
    --left-pad-source False --skip-invalid-size-inputs-valid-test \
    --save-dir checkpoints/$MODEL --tensorboard-logdir logs/$MODEL\
    --seed 1 --memory-efficient --no-epoch-checkpoints --no-progress-bar --log-interval 10 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 \
    --max-tokens 600 --update-freq 14 --max-update 50000 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --lr 0.002 \
    --min-lr '1e-9' --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --convnet resnet --conv-bias --num-layers 14 --kernel-size 11  \
    --aggregator gated-max --add-positional-embeddings --share-decoder-input-output-embed
```

# Evaluate on the test set:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    -s de -t en --gen-subset test \
    --path checkpoints/pa_iwslt_de_en/checkpoint_best.pt \
    --model-overrides "{'max_source_positions': 1024, 'max_target_positions': 1024}" --left-pad-source False  \
    --user-dir examples/pervasive --no-progress-bar \
    --max-tokens 8000 --beam 5 --remove-bpe 
```

# Wait-k decoding with Pervasive Attention

# Training Wait-k Pervasive Attention for IWSLT'14 De-En:
```bash
k=7
MODEL="pa_wait${k}_iwslt_deen"
mkdir -p checkpoints/$MODEL
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en -s de -t en \
    --user-dir examples/pervasive --arch pervasive \
    --max-source-positions 100  --max-target-positions 100 \
    --left-pad-source False --skip-invalid-size-inputs-valid-test \
    --save-dir checkpoints/$MODEL --tensorboard-logdir logs/$MODEL \
    --seed 1 --memory-efficient --no-epoch-checkpoints --no-progress-bar --log-interval 10  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 \
    --max-tokens 600 --update-freq 14 --max-update 50000 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --lr 0.002 \
    --min-lr '1e-9' --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --convnet resnet --conv-bias --num-layers 14 --kernel-size 11  \
    --add-positional-embeddings --share-decoder-input-output-embed \
    --aggregator path-gated-max --waitk $k --unidirectional
```

# Evaluate on the test set:
```bash
k=5 # Evaluation time k
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    -s de -t en --gen-subset test \
    --path checkpoints/pa_wait7_iwslt_deen/checkpoint_best.pt --task waitk_translation --eval-waitk $k \
    --model-overrides "{'max_source_positions': 1024, 'max_target_positions': 1024}" --left-pad-source False  \
    --user-dir examples/pervasive --no-progress-bar \
    --max-tokens 8000 --remove-bpe --beam 1
```

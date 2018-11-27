<<<<<<< HEAD
## Learning Deep Representations of Appearance and Motion for Anomalous Event Detection

    Reimplementation of this paper.
- Dataset UCSD_Anomaly_Dataset

### Usage

```bash
    # pretrain and finetuning
    python ./dae.py 
        --datasetPath "path to dataset"
        --num_epoch "number of epoch(default: 10)"
        --batch_size "batch size(default: 10)"
        --max "max number of dataset per epoch(0 represents all)"
        --corrupt_prob "corrupted data ratio"
        --dimensions "dimensions of hidden layers (default:[1024, 512, 256, 128]"
        --momentum "learning momentum(default:0.9)"
    # evaluation
    python ./eval.py
        --checkpoint_dir "loading latest checkpoint"

    # and so on

```

> Thanks to original work but it is incomplete:[anomaly-event-detection](https://github.com/nabulago/anomaly-event-detection)

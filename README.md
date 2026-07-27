# RLHF Reward Model Scaffold (AI Tutor)

**Archived — a guided-exercise scaffold kept for reference.** The training run does not complete; see Status before drawing conclusions from it.

Sets up training for a reward model — the component that scores response quality in an RLHF pipeline — so that low-quality or misleading tutor explanations could be systematically down-ranked.

## What it configures

Uses the `trl-lib/ultrafeedback_binarized` preference dataset, which pairs a chosen and a rejected response with human-assigned scores. The prompt is extracted from each chosen interaction, a chat template is applied, and the data is tokenized and split 90/10 into train and eval.

Training runs through TRL's `RewardTrainer` with `AutoModelForSequenceClassification` (`num_labels=1`) on top of `openai-gpt`, for 5 epochs at batch size 16, with a `TrainerCallback` that collects per-step loss and plots it at the end.

## Status

**The training cell does not complete.** It fails at `load_dataset` with `NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported` — an environment issue in the hosted notebook rather than a flaw in the training code. As a result there is no trained reward model, no loss curve, and no ranking examples in this repository.

## Known limitations

**The dataset split is the wrong one.** It loads `split='test'` (1000 rows) rather than the 62,000-row train split, so even on a successful run the model would train on the evaluation partition.

**Tokenization and training lengths disagree.** The tokenizer pads to `max_length=4096` while `RewardConfig` sets `max_length=60`, so almost all of each padded sequence is discarded downstream. One of the two numbers is wrong.

**The base model is `openai-gpt`.** That is GPT-1, a 2018 model, specified by the original exercise. Any serious reward model would start from something considerably more capable.

**No evaluation metric.** Even with training working, nothing measures preference accuracy — the rate at which the model scores the chosen response above the rejected one — which is the only number that would say whether the reward model works.

## To make it run

Load the dataset in a local environment rather than the hosted one, switch to `split='train'`, reconcile the two `max_length` values, and add a preference-accuracy metric on the held-out split.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers trl datasets scikit-learn matplotlib jupyter
jupyter notebook notebook.ipynb
```

Requires Python 3.10 and a GPU.

## Stack

Python · PyTorch · Hugging Face Transformers, TRL, Datasets · scikit-learn · matplotlib

## License

MIT

---

*Originated as a guided DataCamp practical exercise, independently re-packaged and documented. The framing in the notebook is illustrative, not a real client engagement.*

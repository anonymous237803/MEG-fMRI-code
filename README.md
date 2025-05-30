# Transformer-based MEG-fMRI Encoding Model with High Spatial and Temporal Resolution

Project Website: [https://anonymous237803.github.io/NeurIPS2025-MEG-fMRI/](https://anonymous237803.github.io/NeurIPS2025-MEG-fMRI/)

## Environment Setup

```{bash}
conda env create -f environment.yml
```

> Note: Make sure you install pytorch with regard to your machine's CUDA version.

## Dataset

Please download our MEG and fMRI dataset from [Google Drive](https://drive.google.com/file/d/1DuQZTUa4Ngc9OhyxglI9rGvb7QoaPnyG/view?usp=sharing). Change `DATASET_DIR` in config files and notebooks to your own dataset path.

## Demo

Run the notebooks to check the results of our trained models.

- `3_evaluate_predict.ipynb`: Predictive performance of our model (Section 3.1).
- `3_ridge_MEG.ipynb`: Predictive performance of MEG ridge ceiling (Section 3.1).
- `3_ridge_fmri.ipynb`: Predictive performance of fMRI ridge ceiling (Section 3.1).
- `5_word_onset_freqnoun.ipynb`: Word-locked cascades of source activities after word onset (Section 4.1).
- `6_surprisal.ipynb`: Word surprisal effect in sources (Section 4.2).
- `6_surprisal_predmeg.ipynb`: Word surprisal effect in predicted MEG sensors (Section 4.2).

## Train

Set the `name` variable in `train.py` to the name of your config file name. Then run the training script:

```
python train.py
```

Uncomment the last few lines to save your trained models into `\trained_models`.

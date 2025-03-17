# KPFBNC

## Datasets
### ABIDE
The ABIDE dataset is openly accessible to all users. You can download it by following the instructions on the [ABIDE website](http://preprocessed-connectomes-project.org/abide/download.html), or with the following code:
   -    from nilearn import datasets
   -        datasets.fetch_abide_pcp(path,band_pass_filtering=True,global_signal_regression=True,quality_checked=True,derivatives='rois_cc200')

### ADHD
The ADHD dataset can be downloaded from the [ADHD website](http://preprocessed-connectomes-project.org/adhd200/download.html).

## Command
python main.py --data_name 'ABIDE200' --denoise True --fisher_transform True --training_strategy 'sd'
options:
  --data_name   'ABIDE200', 'ADHD200'
  --denoise   True, False
  --fisher_transform   True, False
  --training_strategy   'sd', 'bd', 'cd' (represent same dataset, both datasets, and cross dataset respectively)
  
## Requirement
- torch                   2.1.2+cu118
- torch_geometric         2.5.3
- numpy                   1.23.0
- pandas                  2.0.3
- scikit-learn            1.3.2
- nilearn                 0.10.2



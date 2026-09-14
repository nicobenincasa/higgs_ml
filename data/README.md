# Dataset

This project uses the HIGGS dataset for binary classification of
simulated particle-physics events into signal and background.

The dataset is not included in this Git repository because of its size.

## Expected file

Place the following file in this directory:

```text
training.csv
```

The analysis expects the 250,000-event training sample used in the
Higgs Boson Machine Learning Challenge.

The dataset contains:

* 30 physics input features
* `EventId`
* `Weight`
* `Label`

The label is:

* `s` — signal
* `b` — background

See the main project README for details on the analysis.

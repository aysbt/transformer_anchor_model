
````markdown
# Nuclear Mass Excess Prediction Pipeline

## Overview

This project predicts nuclear mass excess using three connected stages:

1. A least-squares nuclear mass formula.
2. A local residual anchor based on nearby known nuclei.
3. A Transformer neural network that predicts the remaining correction.

The project uses a chronological evaluation method:

- Nuclei measured in AME2016 form the training and validation pool.
- Nuclei newly measured in AME2020 form the chronological test set.

This is intended to simulate a realistic scientific prediction problem. The model is trained using information available in AME2016 and is then evaluated on measurements that became available in AME2020.

---

## Main Prediction Idea

The neural network does not predict the full nuclear mass excess directly.

The full prediction is divided into three stages:

```text
Experimental mass excess
        =
Least-squares mass formula
        +
Local residual anchor
        +
Transformer correction
```

In the code, this relationship is:

```python
offset = LSMF + anchor

residual = experimental_mass_excess - offset

predicted_mass_excess = offset + predicted_residual
```

The neural network therefore learns only the remaining residual after the physics baseline and local anchor have explained part of the mass surface.

---

## Complete Project Workflow

```text
Raw AME2016 and AME2020 mass tables
                    │
                    ▼
         Parse and validate data
                    │
                    ▼
 Generate physics features and datasets
                    │
                    ▼
 Fit least-squares baseline and local anchor
                    │
                    ▼
 Normalize features and create PyTorch batches
                    │
                    ▼
          Train Transformer model
                    │
                    ▼
 Evaluate validation and chronological test sets
                    │
                    ▼
 Repeat training with multiple random seeds
                    │
                    ▼
 Aggregate metrics, plots, and tables
```

---

## Project Files

```text
.
├── ame_parser.py
├── anchor.py
├── create_configs.py
├── dataloader.py
├── evaluate.py
├── model.py
├── prepare_data.py
├── run_multiseed.py
├── test.py
├── train.py
└── trainer.py
```

Each file has a separate responsibility in the pipeline.

---

## 1. AME Data Parsing

### File: `ame_parser.py`

This file reads the raw Atomic Mass Evaluation mass tables.

The AME files use a fixed-width text format. This means that each value appears within a specific range of character positions rather than being separated by commas.

The parser extracts the following values:

```text
N                 Neutron number
Z                 Proton number
A                 Mass number, where A = N + Z
mass_excess_keV   Nuclear mass excess in keV
extrapolated      Whether the AME value was marked with #
```

### Extrapolated Values

AME values marked with `#` are extrapolated values rather than direct measurements.

By default, these values are removed.

This is important because including them would mean training the model on values that were already estimated by another model rather than measured experimentally.

### Parser Checks

The parser removes:

- Header lines.
- Comment lines.
- Invalid rows.
- Rows where `A != N + Z`.
- Rows where `A <= 0`.
- Rows without a valid mass-excess value.
- Duplicate `(N, Z)` entries.
- Extrapolated values, unless explicitly requested.

### Validation Nuclei

The parser validates the extracted values using several well-known nuclei:

```text
12C
4He
56Fe
208Pb
238U
```

These checks help identify problems such as:

- Incorrect fixed-width column positions.
- Incorrect AME file versions.
- Incorrect units.
- Parsing the wrong numerical field.

### Chronological Dataset Construction

The parser supports the following chronological split:

```text
Training and validation pool:
Nuclei measured in both AME2016 and AME2020.

Chronological test set:
Nuclei measured in AME2020 but not measured in AME2016.
```

When a nucleus appears in both evaluations, the AME2020 mass value is used because the newer evaluation supersedes the older one.

---

## 2. Data Preparation

### File: `prepare_data.py`

This file converts the raw AME tables into CSV files used by the rest of the pipeline.

It produces:

```text
data/trainval.csv
data/test.csv
```

### Default Command

```bash
python prepare_data.py
```

### Custom Minimum Neutron and Proton Number

The default restriction is:

```text
N >= 8
Z >= 8
```

A different restriction can be provided with:

```bash
python prepare_data.py --min-nz 0
```

For example, `--min-nz 0` disables the default lower cut.

---

## 3. Generated Datasets

### `data/trainval.csv`

This file contains nuclei that were already measured in AME2016.

It is later divided into:

- A training split.
- A validation split.

### `data/test.csv`

This file contains nuclei that were newly measured in AME2020.

These nuclei are not part of the AME2016 training and validation pool.

---

## 4. Generated Physics Features

The `prepare_data.py` script creates all physics features used by the different model configurations.

### Basic Nuclear Features

```text
N
Z
A
```

where:

```text
A = N + Z
```

### Liquid-Drop-Inspired Features

```text
A^2/3
Z(Z-1)/A^1/3
(N-Z)^2/A
```

These approximately represent:

- Nuclear surface effects.
- Coulomb repulsion between protons.
- Neutron-proton asymmetry.

### Asymmetry Features

```text
(N-Z)/A
N/Z
```

These describe the balance between neutron and proton numbers.

### Even-Odd Features

```text
ZEO
NEO
```

These features indicate whether the proton and neutron numbers are even or odd.

The values are calculated as:

```python
ZEO = Z % 2
NEO = N % 2
```

Therefore:

```text
0 = even
1 = odd
```

### Shell Features

```text
Zshell_category
Nshell_category
deltaZ
deltaN
```

The shell categories describe which interval between magic numbers contains the proton or neutron number.

The magic-number list is:

```text
2, 8, 20, 28, 50, 82, 126, 184
```

The `deltaZ` and `deltaN` features give the distance to the nearest magic number.

### Valence Features

```text
proton_particles
proton_holes
neutron_particles
neutron_holes
nu_Z
nu_N
promiscuity
```

These features describe particles and holes relative to nearby closed shells.

The proton and neutron valence values are:

```text
nu_Z = minimum of proton particles and proton holes
nu_N = minimum of neutron particles and neutron holes
```

The promiscuity factor is:

```text
promiscuity = nu_Z × nu_N / (nu_Z + nu_N)
```

When the denominator is zero, the value is set to zero.

### Neighbour-Existence Features

```text
neighbor_N_plus_1_exists
neighbor_N_minus_1_exists
neighbor_Z_plus_1_exists
neighbor_Z_minus_1_exists
```

These columns indicate whether immediate neighbouring nuclei exist in the historical dataset.

Only neighbour-existence flags are stored in the CSV files.

The neighbouring mass values themselves are not stored in the CSV. They are handled by `anchor.py`, where the allowed information source can be controlled explicitly.

---

## 5. Physics Baseline and Local Anchor

### File: `anchor.py`

This file implements the first two stages of the prediction system:

```text
Stage 1: Least-squares mass formula
Stage 2: Local residual anchor
```

---

## 6. Stage 1: Least-Squares Mass Formula

### Class: `LSMFBaseline`

The least-squares mass formula provides a smooth physics-shaped baseline.

A least-squares model finds coefficients that minimize the difference between the formula predictions and the training measurements.

The design matrix contains terms related to:

```text
Nuclear size
Surface effects
Coulomb repulsion
Neutron-proton asymmetry
Pairing effects
Additional smooth corrections
```

The design terms are:

```text
Constant
A
A^(2/3)
Z(Z-1)/A^(1/3)
(N-Z)^2/A
Pairing term / sqrt(A)
A^(1/3)
(N-Z)/A
```

The baseline coefficients are learned only from the training split.

No coefficients are imported from external mass models such as:

```text
FRDM
WS4
HFB
```

### Baseline Fitting

```python
baseline.fit(N, Z, mass_excess)
```

### Baseline Prediction

```python
baseline.predict(N, Z)
```

The output is the baseline mass-excess prediction in keV.

---

## 7. Stage 2: Local Residual Anchor

### Class: `LocalResidualAnchor`

After fitting the baseline, each known nucleus has a residual:

```text
baseline residual =
experimental mass excess - baseline prediction
```

The local anchor estimates this residual at a requested nucleus by examining nearby known nuclei on the `(N, Z)` grid.

A simple analogy is estimating the temperature at an unmeasured location using nearby weather stations.

### Neighbour Search

The anchor searches neighbouring nuclei within a configurable maximum radius.

The default training configuration is:

```text
Maximum radius: 3
Minimum neighbours: 14
```

The search expands one ring at a time until:

- Enough neighbours are found, or
- The maximum radius is reached.

### Self-Label Protection

The exact target location is always excluded:

```python
if dn == 0 and dz == 0:
    continue
```

This prevents a nucleus from reading its own known mass.

Removing this condition would create direct target leakage.

### Local Linear Fit

The local anchor fits nearby residuals using:

```text
Constant term
Neutron displacement, dN
Proton displacement, dZ
Neutron parity mismatch
Proton parity mismatch
```

The estimated anchor is the fitted value at:

```text
dN = 0
dZ = 0
Parity matched
```

### Distance Weighting

Closer neighbours receive larger weights.

The weights decrease with distance using a Gaussian-like expression.

This gives nearby nuclei more influence than distant nuclei.

### Anchor Diagnostic Features

The anchor returns:

```text
anchor
anchor_has
anchor_n
anchor_mean_dist
anchor_max_dist
anchor_scatter
anchor_parity_match
```

Their meanings are:

| Feature | Meaning |
|---|---|
| `anchor` | Estimated local residual in keV |
| `anchor_has` | Whether at least one allowed neighbour was found |
| `anchor_n` | Number of neighbours used |
| `anchor_mean_dist` | Average distance of the neighbours |
| `anchor_max_dist` | Maximum neighbour distance |
| `anchor_scatter` | Weighted local fitting error |
| `anchor_parity_match` | Fraction of neighbours with matching neutron and proton parity |

These values are also available as neural-network input features.

---

## 8. Anchor Context and Information Regimes

### Class: `AnchorContext`

`AnchorContext` combines:

```text
The fitted least-squares baseline
The local residual anchor
The allowed known-mass dictionary
The name of the information regime
```

The supported regimes are:

```text
train
trainval
historical
loo
```

### `train` Regime

```text
Dictionary = training split only
```

This is used during model training.

The target nucleus is excluded from its own local estimate.

### `trainval` Regime

```text
Dictionary = training and validation pool
```

This uses the full AME2016 pool after the training and validation data are combined.

### `historical` Regime

```text
Dictionary = AME2016 measured nuclei only
```

This is the main prospective test regime.

It represents the information that was available before the new AME2020 measurements were known.

### `loo` Regime

```text
Dictionary = AME2016 nuclei and test nuclei, except the target itself
```

`loo` means leave-one-out.

In this regime, test nuclei can support one another.

This is an interpolation experiment rather than a true prospective prediction.

It should be reported separately and should not be presented as the main extrapolation result.

---

## 9. Dataset Loading and Preprocessing

### File: `dataloader.py`

This file converts prepared CSV rows into tensors that PyTorch can use.

A tensor is a numerical array used by PyTorch for neural-network calculations.

The main dataset class is:

```python
NuclearMassDataset
```

### Main Responsibilities

The data loader performs the following steps:

```text
1. Read the CSV file.
2. Create the training or validation split.
3. Separate categorical and continuous features.
4. Construct or receive the anchor context.
5. Calculate the offset.
6. Calculate the anchored residual.
7. Calculate training normalization statistics.
8. Encode categorical values as integer indices.
9. Convert the values into PyTorch tensors.
10. Create data batches.
```

---

## 10. Training and Validation Split

The training and validation data are divided using a stratified split.

The nuclei are grouped into approximate regions using:

```python
bin = f"{int(N / 5)}_{int(Z / 5)}"
```

The split is:

```text
70% training
30% validation
```

The same random seed reproduces the same split.

Bins containing fewer than two nuclei are removed because they cannot be divided between training and validation while preserving stratification.

---

## 11. Continuous and Categorical Features

The current categorical features are:

```text
Zshell_category
Nshell_category
```

All other selected features are treated as continuous features.

The data loader always ensures that the following are present as continuous features:

```text
N
Z
```

even when they are not explicitly included in the selected feature list.

---

## 12. Anchored Residual Target

The offset is:

```text
offset = LSMF prediction + local anchor
```

The residual is:

```text
residual = measured mass excess - offset
```

The neural-network target is the standardized residual:

```text
standardized target =
(residual - training residual mean)
/
training residual standard deviation
```

In code:

```python
std_target = (residual - target_mean) / target_std
```

The neural network predicts this standardized value.

---

## 13. Feature Normalization

Continuous features are standardized using statistics calculated from the training split:

```text
standardized feature =
(feature - training mean)
/
training standard deviation
```

The training means and standard deviations are reused for validation and testing.

Validation and test data must not calculate their own statistics.

If a continuous feature has zero standard deviation, its standard deviation is replaced with `1.0` to avoid division by zero.

Missing continuous values are replaced with the corresponding training mean.

---

## 14. Categorical Encoding

Categorical values are converted into integer indices.

For example:

```text
Original categories:
0, 1, 2, 3

Encoded values:
0, 1, 2, 3
```

A reserved index is added for categories that were not observed in the training data.

For example:

```text
Known categories: 0, 1, 2, 3
Unknown index:    4
```

This prevents an index-out-of-range error when validation or test data contain an unseen category.

---

## 15. Data Loader Outputs

During training, each item contains:

```python
categorical_features, continuous_features, target
```

During evaluation, each item also contains:

```python
N, Z, offset
```

The full evaluation output is:

```python
(
    categorical_features,
    continuous_features,
    target,
    N,
    Z,
    offset
)
```

The offset is needed to reconstruct the physical mass excess.

---

## 16. Transformer Model

### File: `model.py`

This file defines the Transformer neural network used to predict the standardized anchored residual.

The model treats every nuclear feature as a token.

In a language Transformer, tokens may represent words.

In this project, tokens represent physical features such as:

```text
N
Z
A
Shell category
Coulomb term
Asymmetry term
Anchor scatter
```

---

## 17. Model Data Flow

```text
Categorical features ──> embedding tokens ─────┐
                                               │
Continuous features ──> continuous tokens ─────┤
                                               ▼
                                  Feature identity embeddings
                                               │
                                               ▼
                                  Transformer encoder layers
                                               │
                                               ▼
                      Separate categorical and continuous pooling
                                               │
                                               ▼
                                         Fusion layer
                                               │
                              ┌────────────────┴────────────────┐
                              ▼                                 ▼
                    Linear continuous baseline       Nonlinear correction
                              │                                 │
                              └────────────────┬────────────────┘
                                               ▼
                              Standardized anchored residual
```

---

## 18. Categorical Embeddings

### Class: `InputEmbeddings`

Each categorical feature has its own embedding table.

An embedding converts an integer category into a learned vector.

For example:

```text
Shell category 3
        ↓
A learned vector with 128 numbers
```

Input shape:

```text
(batch size, number of categorical features)
```

Output shape:

```text
(batch size, number of categorical features, d_model)
```

---

## 19. Continuous Feature Tokenization

### Class: `HybridContinuousFeatureTokenizer`

Each continuous feature is converted into a vector token.

The tokenizer combines:

1. A shared linear projection.
2. A feature-specific correction.

The shared component is:

```text
shared token = shared Linear(1 → d_model)(feature value)
```

The feature-specific component is:

```text
feature correction =
feature value × feature-specific weight
+ feature-specific bias
```

The final token is:

```text
token = shared token + feature correction
```

The feature-specific correction starts near zero.

This means the model begins close to the simpler shared representation and gradually learns feature-specific behavior during training.

---

## 20. Shell Value Encoding

### Class: `ValueBasedPositionalEncoding`

The following categorical features receive a special encoding:

```text
Zshell_category
Nshell_category
```

These categories are not treated as completely unrelated labels.

Their numerical values represent ordered shell regions.

A sinusoidal encoding is therefore added based on the shell-category value.

This is different from ordinary sequence positional encoding.

Ordinary positional encoding represents:

```text
Token position 0
Token position 1
Token position 2
```

Shell value encoding represents:

```text
Shell category value 0
Shell category value 1
Shell category value 2
```

---

## 21. Feature Identity Embeddings

Each feature receives a learned identity embedding.

This tells the Transformer which physical feature each token represents.

For example:

```text
Token 0 = Zshell_category
Token 1 = Nshell_category
Token 2 = N
Token 3 = Z
Token 4 = A
```

This is important because nuclear features are not a natural word sequence.

The feature identity embedding allows the model to distinguish two features even when they have similar numerical values.

---

## 22. Multi-Head Self-Attention

### Class: `MultiHeadAttentionBlock`

Self-attention allows every feature token to examine every other feature token.

For example, the model may learn relationships between:

```text
N and Z
Shell category and valence count
Coulomb term and proton number
Anchor scatter and local neighbour count
```

The model calculates:

```text
Query vectors
Key vectors
Value vectors
```

Attention scores are calculated using:

```text
score = Query × Key / sqrt(head dimension)
```

A softmax operation converts the scores into normalized attention weights.

The weighted values are then combined into updated feature representations.

The model uses multiple attention heads so that different heads can learn different relationships.

---

## 23. Transformer Encoder

### Classes

```text
EncoderBlock
TransformerEncoder
```

Each encoder block contains:

```text
Layer normalization
Multi-head self-attention
Residual connection
Feed-forward network
Second residual connection
```

The encoder uses a pre-normalization structure:

```text
x = x + attention(normalize(x))
x = x + feed_forward(normalize(x))
```

This is generally more stable for deeper Transformer models.

The complete encoder stacks several encoder blocks.

Attention weights are stored for later visualization.

---

## 24. Pooling

After the Transformer processes all tokens, the token information must be combined into fixed-size vectors.

The model supports:

```text
mean
attention
gated_attention
```

### Mean Pooling

```text
pooled vector = average of all token vectors
```

### Attention Pooling

The model learns a score for every token.

Softmax converts the scores into weights.

The pooled vector is:

```text
pooled vector =
sum of token weight × token vector
```

### Gated Attention Pooling

The current default is gated attention.

It combines mean pooling and attention pooling:

```text
pooled =
(1 - gate) × mean pooled
+
gate × attention pooled
```

The initial gate value is approximately:

```text
sigmoid(-2) ≈ 0.12
```

Therefore, training begins approximately as:

```text
88% mean pooling
12% attention pooling
```

The gate can change during training.

---

## 25. Separate Feature Branch Pooling

Categorical and continuous tokens are pooled separately.

The model produces:

```text
pooled categorical representation
pooled continuous representation
```

These representations are concatenated:

```text
fused =
[pooled categorical, pooled continuous]
```

When there are no categorical features, the categorical pooled vector is replaced with zeros.

---

## 26. Final Regression

The model contains two prediction paths.

### Linear Baseline

```python
baseline = baseline_layer(continuous_inputs)
```

This learns a simple linear relationship from the continuous inputs.

### Nonlinear Transformer Correction

```python
correction = regressor(fused)
```

This uses the Transformer representation.

### Final Model Output

```python
prediction = baseline + correction
```

Despite the variable name in the model code, this value represents the standardized anchored residual, not the final physical mass excess.

---

## 27. Default Model Architecture

The default architecture is defined in `train.py`:

```python
ARCH = {
    "d_model": 128,
    "num_heads": 8,
    "d_ff": 512,
    "num_layers": 4,
    "dropout": 0.1,
    "pooling_type": "gated_attention",
}
```

This means:

| Parameter | Value |
|---|---:|
| Token dimension | 128 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Feed-forward size | 512 |
| Dropout | 0.1 |
| Pooling | Gated attention |

---

## 28. Model Feature Sets

### File: `train.py`

The training script defines several feature configurations.

### `CoreModel`

```text
N
Z
A
```

### `ShellModel`

```text
N
Z
A
Zshell_category
Nshell_category
```

### `ZEOModel`

```text
N
Z
A
ZEO
NEO
```

### `MagicModel`

```text
N
Z
A
deltaN
deltaZ
```

### `LiquidDropModel`

```text
N
Z
A
A^2/3
Z(Z-1)/A^1/3
(N-Z)^2/A
```

### `ValenceModel`

```text
N
Z
A
nu_N
nu_Z
promiscuity
proton_particles
proton_holes
neutron_particles
neutron_holes
```

### `AnchoredFullModel`

This model uses all features in the complete physics feature list.

The anchor diagnostic features are added automatically by the data loader when anchoring is enabled.

---

## 29. Training Orchestration

### File: `train.py`

This is the main script for training one model with one random seed.

Example:

```bash
python train.py --model AnchoredFullModel --seed 42
```

The script performs the following operations:

```text
1. Set the random seed.
2. Select the model feature set.
3. Build the training data loader.
4. Fit the least-squares baseline.
5. Build the training anchor dictionary.
6. Save preprocessing statistics.
7. Build validation data loaders.
8. Create the Transformer model.
9. Train the model.
10. Reload the best checkpoint.
11. Evaluate the validation set.
12. Save metrics, predictions, and plots.
```

---

## 30. Saved Training Files

The training script saves the neural-network weights:

```text
model_checkpoint/best_<experiment>.pth
```

It also saves the preprocessing and anchor information:

```text
model_checkpoint/preprocess_<experiment>.pth
```

The preprocessing bundle contains:

```text
Training normalization statistics
Anchor state
Selected feature list
Continuous feature order
Categorical feature order
Categorical feature sizes
Model architecture
Whether anchoring was enabled
```

Both files are required for consistent testing.

---

## 31. Training Loop

### File: `trainer.py`

The training loop repeats the following process for every epoch:

```text
1. Send each training batch through the model.
2. Calculate the loss.
3. Calculate gradients.
4. Clip large gradients.
5. Update model parameters.
6. Evaluate the validation set.
7. Update the learning-rate scheduler.
8. Save the best checkpoint.
9. Check the early-stopping condition.
```

An epoch is one complete pass through the training dataset.

---

## 32. Loss Function

The project uses:

```python
torch.nn.SmoothL1Loss(beta=0.25)
```

Smooth L1 loss behaves approximately like:

- Squared error for small errors.
- Absolute error for large errors.

This makes it less sensitive to very large outliers than ordinary mean squared error.

---

## 33. Optimizer

The project uses:

```python
AdamW
```

with:

```text
Default learning rate: 0.001
Weight decay: 0.0001
```

The optimizer updates the model parameters using the calculated gradients.

---

## 34. Gradient Clipping

Gradients are limited using:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

This prevents extremely large gradients from causing unstable parameter updates.

---

## 35. Learning-Rate Scheduler

The project uses:

```python
ReduceLROnPlateau
```

The learning rate is reduced when the smoothed validation loss stops improving.

The configuration is:

```text
Reduction factor: 0.5
Scheduler patience: 6 epochs
```

---

## 36. Early Stopping

Early stopping uses an exponentially smoothed validation loss.

The smoothed loss is:

```text
EMA =
alpha × current validation loss
+
(1 - alpha) × previous EMA
```

The default smoothing factor is:

```text
alpha = 0.3
```

Training stops when the smoothed validation loss does not improve for the configured patience period.

The default patience is:

```text
30 epochs
```

---

## 37. Training Metrics in keV

The neural network operates in standardized residual units.

However, the training loop reports RMSE and MAE in keV.

Because the offset appears in both the prediction and target, it cancels when calculating the error:

```text
prediction error in keV =
(predicted standardized residual
-
actual standardized residual)
× target standard deviation
```

This allows physical-unit metrics to be calculated during every epoch.

---

## 38. Evaluation

### File: `evaluate.py`

This file contains shared functions for model inference, metrics, regional analysis, and plots.

### Physical Reconstruction

The network prediction is converted back to physical units using:

```text
predicted mass excess =
predicted standardized residual
× target standard deviation
+ target mean
+ offset
```

In code:

```python
predicted_mass = (
    predicted_standardized_residual * target_std
    + target_mean
    + offset
)
```

The actual target is reconstructed using the same transformation.

No final metric should be calculated in standardized units.

---

## 39. Evaluation Metrics

The following metrics are calculated:

```text
Number of nuclei
RMSE
MAE
Mean signed error
Median absolute error
90th percentile absolute error
Maximum absolute error
Fraction within 100 keV
Fraction within 250 keV
Fraction within 500 keV
```

### RMSE

Root mean squared error is:

```text
RMSE = sqrt(mean((prediction - measurement)^2))
```

RMSE gives more weight to large errors.

### MAE

Mean absolute error is:

```text
MAE = mean(abs(prediction - measurement))
```

MAE measures the average absolute prediction error.

### Bias

Mean signed error is:

```text
bias = mean(prediction - measurement)
```

Interpretation:

```text
Positive bias:
Predictions are too high on average.

Negative bias:
Predictions are too low on average.

Bias near zero:
No strong average direction.
```

---

## 40. Regional Metrics

Nuclei are divided into broad regions.

### Very Light

```text
A < 20
```

### Light

```text
20 <= A < 60
```

### Medium

```text
60 <= A <= 200
```

### Heavy

```text
A > 200
```

### Heavy Exotic

```text
Z > 100 and N > 150
```

Regional RMSE, MAE, bias, and maximum error are calculated separately.

---

## 41. Generated Plots

The evaluation code can generate:

```text
Training and validation loss curves
Training and validation RMSE curves
Signed error map on the N-Z plane
Error distribution histogram
Three-dimensional prediction plot
Predicted-versus-experimental plot
Stage-by-stage RMSE plot
Attention-weight heatmap
Attention-based feature score table
```

Generated plots are stored in:

```text
plots/
```

Generated numerical results are stored in:

```text
results/
```

---

## 42. Chronological Testing

### File: `test.py`

This script loads a trained model and evaluates it on the nuclei newly measured in AME2020.

Example:

```bash
python test.py --model AnchoredFullModel --seed 42
```

The same neural-network weights are used for every anchor regime.

Only the anchor dictionary changes.

---

## 43. Test Anchor Regimes

### Historical

```text
Dictionary = AME2016 measured nuclei
```

This is the main prospective result.

It uses only the mass information available before the AME2020 measurements.

### Train

```text
Dictionary = training split only
```

This is more restrictive than using the full AME2016 pool.

### Trainval

```text
Dictionary = training and validation pool
```

For this dataset, this normally corresponds to the full AME2016 pool.

### Leave-One-Out

```text
Dictionary =
AME2016 nuclei + test nuclei,
excluding the target itself
```

This allows test nuclei to help predict one another.

It is an interpolation result and should not be reported as the main prospective prediction result.

---

## 44. Test Overlap Check

Before testing, the script checks whether any chronological test nucleus appears in the AME2016 pool.

The required overlap is:

```text
0
```

Any nonzero overlap would indicate a problem in the chronological split.

---

## 45. Stage Decomposition

The test script evaluates three prediction stages.

### Stage 1

```text
LSMF only
```

### Stage 1 + Stage 2

```text
LSMF + local anchor
```

### Stage 1 + Stage 2 + Stage 3

```text
LSMF + local anchor + Transformer
```

This decomposition shows how much each stage contributes to the final result.

The neural-network contribution is calculated as:

```text
network contribution =
anchor-only RMSE - full-model RMSE
```

A small network contribution would indicate that most of the performance comes from the baseline and local interpolation.

---

## 46. Anchor Availability Analysis

The test script separately examines nuclei:

```text
With an available anchor
Without an available anchor
```

When no anchor is available, the offset contains only the least-squares baseline.

This helps distinguish local interpolation performance from more independent machine-learning prediction.

---

## 47. Multiple-Seed Experiments

### File: `run_multiseed.py`

A single neural-network run may be unusually good or unusually bad because of random variation.

The random seed affects:

```text
Network parameter initialization
Training batch order
Training-validation split
```

The chronological test set does not change with the seed.

### Default Seeds

```text
17
33
42
67
89
```

### Example Command

```bash
python run_multiseed.py \
    --model AnchoredFullModel \
    --seeds 17,33,42,67,89 \
    --num_epochs 200
```

Each seed is launched as a separate process.

This prevents random states, cached tensors, or global PyTorch state from carrying between runs.

---

## 48. Multi-Seed Outputs

The script aggregates results into:

```text
results/summary_val_<model>.csv
results/summary_test_<model>.csv
results/summary_regions_<model>.csv
results/latex_tables_<model>.txt
plots/<model>_seed_spread.png
```

The summary files contain:

```text
Mean performance
Standard deviation between seeds
Validation results
Test results by anchor regime
Regional results
Stage-decomposition results
```

The LaTeX output contains tables that can be inserted into a scientific paper.

---

## 49. Configuration Export

### File: `create_configs.py`

This file creates readable JSON descriptions of each model configuration.

Run:

```bash
python create_configs.py
```

Example output:

```text
model_checkpoint/CoreModel_config.json
model_checkpoint/ShellModel_config.json
model_checkpoint/AnchoredFullModel_config.json
```

Each configuration contains:

```text
Model name
Description
Selected feature list
Architecture
Target definition
Preprocessing note
```

Categorical feature sizes are not stored in these JSON files.

They are calculated from the training data and stored in:

```text
model_checkpoint/preprocess_<experiment>.pth
```

This avoids hard-coded embedding sizes and index-out-of-range errors.

---

## 50. Recommended Execution Order

### Step 1: Prepare the Data

```bash
python prepare_data.py
```

This creates:

```text
data/trainval.csv
data/test.csv
```

### Step 2: Create Model Configuration Files

```bash
python create_configs.py
```

This step is optional but useful for documentation.

### Step 3: Train One Model

```bash
python train.py \
    --model AnchoredFullModel \
    --seed 42
```

### Step 4: Test the Trained Model

```bash
python test.py \
    --model AnchoredFullModel \
    --seed 42
```

### Step 5: Run the Multi-Seed Experiment

```bash
python run_multiseed.py \
    --model AnchoredFullModel \
    --seeds 17,33,42,67,89 \
    --num_epochs 200
```

---

## 51. File Dependency Diagram

```text
ame_parser.py
      │
      ▼
prepare_data.py
      │
      ├───────────────> data/trainval.csv
      └───────────────> data/test.csv
                              │
                              ▼
anchor.py <──────────── dataloader.py
                              │
                              ▼
                           model.py
                              │
                              ▼
train.py ───────────────> trainer.py
   │                          │
   └──────────────┬───────────┘
                  ▼
              evaluate.py
                  │
                  ▼
                test.py
                  │
                  ▼
           run_multiseed.py
```

---

## 52. End-to-End Prediction Flow

For one nucleus, the prediction process is:

```text
Input nucleus:
N, Z, A and physics features
            │
            ▼
Least-squares baseline prediction
            │
            ▼
Search allowed nearby known nuclei
            │
            ▼
Estimate local residual anchor
            │
            ▼
offset = baseline + anchor
            │
            ▼
Normalize all neural-network inputs
            │
            ▼
Transformer predicts standardized residual
            │
            ▼
Convert residual back to keV
            │
            ▼
Add offset
            │
            ▼
Final predicted mass excess in keV
```

The final reconstruction is:

```text
predicted mass excess =
LSMF
+
local anchor
+
Transformer residual correction
```

---

## 53. Important Leakage Controls

Two different leakage risks are considered.

### Leakage A: Self-Label Leakage

A target nucleus could read its own measured mass through the anchor dictionary.

Protection:

```python
if dn == 0 and dz == 0:
    continue
```

This condition must not be removed.

### Leakage B: Held-Out Nuclei Supporting One Another

A test nucleus could read the measured mass of another test nucleus.

Protection:

- Use the `historical` regime for the main prospective result.
- Use `loo` only as a separately labelled interpolation experiment.

### Preprocessing Leakage

Validation and test statistics must not be calculated independently.

Protection:

- Means and standard deviations are calculated from the training split.
- Validation and test datasets receive the saved training statistics.

---

## 54. Main Interfaces That Require Detailed Verification

The following boundaries should be checked carefully during code review.

### Raw AME Text to Parsed Data

Questions:

- Are the fixed-width positions correct?
- Is the mass-excess column parsed correctly for both AME versions?
- Are values in keV?
- Are extrapolated values removed correctly?

### AME2016 and AME2020 to Chronological Split

Questions:

- Are test nuclei absent from the AME2016 measured pool?
- Are AME2020 values used for common nuclei?
- Is the `N >= 8` and `Z >= 8` restriction intentional?

### Full Dataset to Training and Validation Split

Questions:

- Does the same seed always reproduce the same split?
- Are sparse bins being removed intentionally?
- Is a 30% validation fraction appropriate?

### Training Data to Anchor Dictionary

Questions:

- Is the target always excluded from its own anchor?
- Can validation masses enter the training dictionary?
- Can test masses enter the historical dictionary?
- Is the selected regime recorded correctly?

### Anchor Output to Preprocessing Statistics

Questions:

- Are statistics calculated only from training rows?
- Are validation and test rows transformed using training statistics?
- Are raw anchor flags accidentally standardized before diagnostic use?

### Dataset to Model

Questions:

- Is the feature order preserved?
- Do categorical sizes match the embedding tables?
- Does the feature identity order match the data-loader order?
- Are categorical features listed before continuous features consistently?

### Model Output to Physical Mass Excess

Questions:

- Is the standardized residual converted back exactly once?
- Is the target mean added once?
- Is the target standard deviation applied once?
- Is the offset added once?
- Are all metrics calculated in keV?

### Checkpoint to Test Reconstruction

Questions:

- Is the same architecture restored?
- Are the same preprocessing statistics restored?
- Is the same feature order restored?
- Is the trained least-squares baseline restored?
- Is the correct anchor regime installed?

### Per-Seed Results to Aggregated Results

Questions:

- Are result files from unrelated experiments included?
- Are exactly the requested seeds aggregated?
- Are old files removed or filtered?
- Is standard deviation calculated across independent runs?

---

## 55. Summary

The project is organized into the following major layers:

```text
Data parsing
    ↓
Physics feature generation
    ↓
Chronological dataset construction
    ↓
Least-squares physics baseline
    ↓
Local residual anchor
    ↓
Data normalization and batching
    ↓
Transformer residual prediction
    ↓
Physical-unit reconstruction
    ↓
Validation and chronological testing
    ↓
Multiple-seed aggregation
```

The most important scientific structure is:

```text
Final prediction =
physics baseline
+
local historical estimate
+
neural-network correction
```

The most important result for prospective evaluation is the chronological AME2020 test result using the `historical` anchor regime.

The `loo` regime should be treated as a separate leave-one-out interpolation experiment.
````

# Anchored transformer for nuclear mass-excess prediction

The transformer is unchanged. What changes is **what it is asked to predict**.

```
ΔM(N,Z)  =  LSMF(N,Z)          Stage 1  8-coefficient mass formula, least-squares
          + anchor(N,Z)        Stage 2  local plane fit through measured neighbours
          + transformer(...)   Stage 3  the network, on the anchored residual
```

A controlled experiment on the same architecture, same features, same split and
same seed — only the target differing — gives:

| Target | Val RMSE | Test RMSE |
|---|---:|---:|
| signed-log(mass excess) — direct | 2,261 keV | 5,622 keV |
| anchored residual | **287 keV** | **840 keV** |

The signed-log transform is gone entirely. The anchored residual is already a
small, roughly symmetric quantity, so the whole class of missing-×10000
inverse-transform bugs cannot occur.

---

## 1. Install

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

CPU is fine. One training run is a few minutes.

---

## 2. Prepare the data

```bash
python prepare_data.py
```

Reads `data/mass16.txt` and `data/mass20.txt` (raw AME tables, included) and writes:

- `data/trainval.csv` — 2368 nuclei measured in AME2016 (train + validation pool)
- `data/test.csv` — 88 nuclei **first measured in AME2020** (chronological test)

Every physics feature is generated here: liquid-drop terms, shell categories,
parity, magic-number distances, valence nucleons, Casten–Cakirli promiscuity,
neighbour-existence flags.

Two things this script does deliberately:

- **Drops AME `#`-flagged values.** AME marks a value with `#` when it is *not
  measured* but extrapolated from the mass surface. Training on those means
  training on somebody else's extrapolation and reporting the agreement as
  experimental accuracy.
- **Does not put neighbour mass values in the CSV.** Only existence flags. The
  neighbour masses are handled by `anchor.py`, where the allowed dictionary is
  controlled explicitly and every number stays auditable.

The parser is validated against ¹²C, ⁴He, ⁵⁶Fe, ²⁰⁸Pb and ²³⁸U on every run.

---

## 3. Train one model

```bash
python train.py --model AnchoredFullModel --seed 42 --num_epochs 200
```

Prints, in order: data and anchor summary, per-epoch table with **RMSE in keV**,
validation metrics, stage decomposition, regional table, and saves all figures.

Options:

```
--model      CoreModel | ShellModel | ZEOModel | MagicModel
             | LiquidDropModel | ValenceModel | AnchoredFullModel
--use_anchor 1 (default) or 0   -> ablation: turn anchoring off entirely
--anchor_radius 3               -> neighbour search radius
--anchor_min_neighbors 14
--seed, --num_epochs, --batch_size, --lr, --patience, --verbose_every
```

Writes `model_checkpoint/best_*.pth` and `model_checkpoint/preprocess_*.pth`
(the latter contains the normalisation statistics **and** the fitted anchor
state, so `test.py` reconstructs everything exactly).

---

## 4. Test one model — the four dictionary regimes

```bash
python test.py --model AnchoredFullModel --seed 42
```

The network is identical in all four; only the neighbour masses the anchor may
see change. **This distinction determines what your number means.**

| Regime | Dictionary | Interpretation |
|---|---|---|
| `historical` | AME2016 only | **Prospective. Report this one.** Exactly the information that existed before the AME2020 measurements. |
| `train` | training split only | Slightly more conservative. |
| `trainval` | training + validation | Full AME2016 pool. |
| `loo` | everything except the target | Leave-one-out **interpolation**. Test nuclei support each other → optimistic. Report separately, labelled. |

Also printed: **stage decomposition** (LSMF alone / +anchor / +transformer). If
the network gain is small, the accuracy is coming from interpolation and the
reader deserves to know.

---

## 5. Multi-seed run — this is where the ± comes from

```bash
python run_multiseed.py --seeds 17,33,42,67,89 --num_epochs 200
```

Runs the full train→test cycle once per seed, **each in a separate process** so
no state leaks between runs, then aggregates to mean ± standard deviation.

What varies with the seed: network initialisation, batch ordering, and the
stratified train/validation split. The chronological test set never changes — it
is fixed by the AME2016/AME2020 chronology — so the spread on the test numbers is
pure model variability, which is what should be quoted.

Outputs:

```
results/summary_val_<model>.csv       mean ± std, validation
results/summary_test_<model>.csv      mean ± std, per regime
results/summary_regions_<model>.csv   mean ± std, per region
results/latex_tables_<model>.txt      paste straight into the paper
plots/<model>_seed_spread.png
```

To re-aggregate without retraining:

```bash
python run_multiseed.py --skip_runs
```

---

## 6. Ablation: does the anchor actually do the work?

```bash
python train.py --model AnchoredFullModel --seed 42 --use_anchor 1
python train.py --model AnchoredFullModel --seed 42 --use_anchor 0
```

---

## Leak control — read before modifying `anchor.py`

There are **two** distinct leaks and they need separate defences.

**Leak A — self label.** Nucleus X reads X's own mass.
Defence: `LocalResidualAnchor._neighbors()` always discards the exact
`(dN, dZ) = (0, 0)` match. This makes training anchors leave-one-out *by
construction*, so the anchor feature has the same meaning and the same difficulty
at fit time as at predict time. **Never remove that check.** Without it you get
spectacular, meaningless numbers.

**Leak B — mutual support.** Nucleus X reads neighbour Y, where Y is also held out
and would not have been known at prediction time.
Defence: the caller chooses the dictionary. `anchor.py` never decides for you.
This is why `test.py` reports all four regimes side by side.

Measured consequence on this dataset — the same model, changing only the
dictionary, moves the chronological test RMSE from **710 keV** (historical) to
**533 keV** (leave-one-out). That 177 keV gap *is* Leak B. Both numbers are
legitimate; they answer different questions.

A related warning: a random K-fold CV over the AME2016 pool scatters neighbours
across folds, so nearly every validation nucleus has training neighbours. Blocked
CV that removes a one-nucleus buffer around each held-out tile drops the same
model from ~255 keV to ~2,334 keV. **Random-split CV numbers therefore measure
interpolation between measured nuclei, not extrapolation.** Say so in the text.

---

## How to frame the results

Honest:

> For nuclei with measured neighbours, an anchored transformer reaches
> RMSE ≈ 0.7 MeV on the 88 nuclei first measured in AME2020, using only masses
> known in AME2016 (one-step extrapolation from the measured frontier).

Not honest:

> The model predicts unmeasured exotic nuclei to 0.7 MeV.

The moment a nucleus has no measured neighbours the anchor contributes nothing
and performance falls back to the feature-only model. `test.py` prints the
stratification (`with anchor` / `no anchor`) so this is never hidden.

---

## Files

```
prepare_data.py     raw AME tables -> trainval.csv / test.csv
ame_parser.py       fixed-width AME parser, drops '#' extrapolations, self-validating
anchor.py           Stage 1 LSMF + Stage 2 local anchor + dictionary regimes
dataloader.py       anchored-residual target, train-only statistics
model.py            the transformer (unchanged)
trainer.py          training loop, per-epoch keV reporting, EMA early stopping
evaluate.py         metrics (always keV), regional split, all plots
train.py            train one model / one seed
test.py             test one model, four dictionary regimes
run_multiseed.py    train+test per seed -> mean ± std -> LaTeX tables
```



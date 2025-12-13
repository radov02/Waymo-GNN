# Training Issues Analysis and Fixes

## Current Symptoms
- **Loss**: ~200 (very high, should be < 10)
- **Cosine Similarity**: NEGATIVE (-0.0023 to -0.0033) - predictions point BACKWARDS!
- **Angle Error**: >100° - model predicting opposite direction
- **Displacement Error**: 200-700m - catastrophically wrong
- **Model NOT learning**: loss barely decreasing

## Root Causes Identified

### 1. ✅ FIXED: Feature Normalization (Centimeters vs Meters)
**Problem**: Waymo positions are in centimeters, but weren't converted before normalization
- `rel_x_to_sdc` was 100x too large
- `dist_to_nearest` was 100x too large

**Fix Applied**: Added `/100.0` conversion in `graph_creation_functions.py` lines 323-340

### 2. ✅ FIXED: Velocity Denormalization in Loss
**Problem**: Loss function used wrong denormalization constant
- Used 10.0 instead of 30.0 (MAX_SPEED)

**Fix Applied**: Changed in `helpers.py` lines 23, 45, 46

### 3. ✅ FIXED: Velocity Magnitude Loss Too Strict
**Problem**: Constant velocity assumption invalid for turning/accelerating vehicles

**Fix Applied**: Only apply to low-acceleration agents in `helpers.py`

### 4. ✅ FIXED: Loss Weights Imbalanced  
**Problem**: Too much weight on velocity magnitude, not enough on position

**Fix Applied**: Updated `config.py`:
- `loss_alpha = 0.3` (angle)
- `loss_beta = 0.5` (MSE) - INCREASED
- `loss_gamma = 0.1` (velocity) - REDUCED
- `loss_delta = 0.1` (cosine) - ENABLED

### 5. ✅ NEW FIX: Model Output Not Constrained
**Problem**: Final GCN layer has no activation - can output ANY value
- Unbounded outputs lead to exploding predictions
- Poor initialization of final layer weights

**Fix Applied**: Added to `EvolveGCNH.py`:
1. Tanh activation + scaling on final layer output: `tanh(x) * 3.0`
2. Better initialization (xavier with gain=0.1) for final layer
3. Constrains predictions to [-3m, +3m] range (reasonable for 0.1s)

## Actions Required

### IMMEDIATE (Do These Now):

1. **Regenerate HDF5 files** (fixes normalization):
   ```bash
   python ./src/graph_creation_and_saving.py
   ```

2. **Run diagnostic scripts** (before training):
   ```bash
   python ./src/check_batch.py    # Check if data is correctly normalized
   python ./src/simple_test.py    # Check if model can learn at all
   ```

3. **Restart training** (with fixes):
   ```bash
   python ./src/training.py
   ```

### Expected Results After Fixes:

**Epoch 1-3**:
- Loss: should drop from ~50 to ~10
- Cosine Similarity: should be POSITIVE (0.3-0.6)
- Angle Error: should be < 60°
- Displacement Error: should be < 50m

**Epoch 5-10**:
- Loss: should drop to ~2-5
- Cosine Similarity: should be 0.6-0.8
- Angle Error: should be < 30°
- Displacement Error: should be 5-20m

**Epoch 15-30**:
- Loss: should converge to ~0.5-2
- Cosine Similarity: should be 0.8-0.95
- Angle Error: should be < 15°
- Displacement Error: should be 1-5m

### If Still Having Issues:

Run these debug scripts to identify the problem:
```bash
python ./src/check_batch.py    # Check data quality
python ./src/debug_data.py     # Full data analysis
python ./src/simple_test.py    # Check if model can overfit
```

Common remaining issues:
1. **Data not regenerated**: Old HDF5 files still have wrong normalization
2. **Model not learning at all**: Check simple_test.py results
3. **Negative cosine similarity**: Model predicting backwards - check output signs

## Key Changes Summary

### Files Modified:
1. `src/helper_functions/graph_creation_functions.py` - Fixed cm→m conversion
2. `src/helper_functions/helpers.py` - Fixed denormalization constants, improved velocity loss
3. `src/config.py` - Rebalanced loss weights
4. `src/EvolveGCNH.py` - Added output scaling and better initialization

### Files Created (Diagnostics):
1. `src/check_batch.py` - Quick data check
2. `src/debug_data.py` - Full data analysis
3. `src/simple_test.py` - Model overfitting test

## Technical Details

### Output Scaling Rationale:
For 0.1s timestep:
- Typical vehicle speed: 5-30 m/s (18-108 km/h)
- Displacement in 0.1s: 0.5-3 meters
- Maximum reasonable: ~5 meters (180 km/h)
- Using `tanh * 3.0` gives range [-3, +3] meters
- This is slightly conservative but prevents extreme predictions

### Why Negative Cosine Similarity?
Possible causes:
1. Model weights initialized poorly
2. Loss function gradients pointing wrong direction
3. Data labels have wrong sign
4. Feature scaling causing instability

The tanh output scaling should fix #1 and #4.

## Monitoring Training

Watch for these warning signs:
- **Cosine similarity stays negative**: Model fundamentally broken
- **Loss increases**: Learning rate too high or gradient explosion
- **No improvement after 10 epochs**: Data or model architecture issue
- **NaN loss**: Gradient explosion, reduce learning rate or check data

Good signs:
- **Steady loss decrease**: Model learning
- **Increasing cosine similarity**: Directional predictions improving
- **Decreasing angle error**: Fine-tuning direction accuracy
- **Validation tracks training**: No severe overfitting

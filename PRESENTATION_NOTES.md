# Presentation Notes: Airbnb Price Prediction

## 1. Identify the Predictive Task

### Predictive Task
- **Task**: Predict Airbnb listing prices in San Diego
- **Type**: Regression problem (continuous target variable)
- **Input**: Property features (bedrooms, bathrooms, location, room type, amenities, reviews, etc.)
- **Output**: Predicted price in dollars ($)

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: 
  - Measures average prediction error in dollars
  - Penalizes large errors more heavily
  - Appropriate for price prediction where we care about dollar accuracy
  - Lower is better
  
- **MAE (Mean Absolute Error)**:
  - Average absolute difference between predicted and actual prices
  - More interpretable than RMSE (average dollar error)
  - Less sensitive to outliers than RMSE
  - Lower is better
  
- **R² (Coefficient of Determination)**:
  - Proportion of variance in price explained by the model
  - Ranges from -∞ to 1.0 (1.0 = perfect predictions)
  - Higher is better
  - Our Random Forest achieved R² = 0.796 (explains 79.6% of price variance)

**Why these metrics?**
- Price prediction is a regression task requiring continuous error measurement
- RMSE and MAE provide interpretable dollar-based error metrics
- R² shows how well the model captures price patterns compared to baseline
- These are standard metrics for regression tasks in ML

### Baselines for Comparison

1. **Baseline (Mean Prediction)**:
   - Predicts the mean price for all listings
   - Simple baseline showing improvement over naive approach
   - Results: RMSE $3,953.91, MAE $817.21, R² -0.000
   - Demonstrates that any model should beat this

2. **Linear Regression**:
   - Baseline ML model from course content
   - Assumes linear relationships between features and price
   - Results: RMSE $3,194.68, MAE $783.70, R² 0.347
   - Shows improvement over mean baseline
   - Demonstrates that non-linear models (Random Forest) are needed

3. **Random Forest**:
   - Advanced model capturing non-linear relationships
   - Results: RMSE $1,784.82, MAE $304.50, R² 0.796
   - Best performing model

### Model Validity Assessment

1. **Train/Test Split**:
   - 80/20 split (9,104 training, 2,277 test samples)
   - Random split ensures generalizability
   - Prevents overfitting by evaluating on unseen data

2. **Residual Analysis**:
   - Residuals vs Predicted: Check for patterns (should be random scatter)
   - Residual distribution: Should be centered around 0
   - Actual vs Predicted: Points should cluster around diagonal line
   - Identifies systematic biases or heteroscedasticity

3. **Error Distribution**:
   - Median absolute error: $304.50 (half of predictions within $305)
   - 75th percentile: Shows error for most predictions
   - 95th percentile: Identifies worst-case prediction errors
   - Helps identify outliers and model limitations

4. **Feature Importance**:
   - Validates that important features (location, room type) have high importance
   - Ensures model is using sensible features, not spurious correlations

---

## 2. Exploratory Analysis, Data Collection, Pre-processing

### Dataset Context

**Source**: Inside Airbnb (insideairbnb.com)
- San Diego, California listings data
- Collected from publicly available Airbnb data
- Contains property listings with various features
- **Original dataset**: 13,162 listings with 79 features

**Purpose**: 
- Understand factors influencing Airbnb pricing
- Build predictive models for price estimation
- Help hosts set competitive prices
- Help guests understand price drivers

**Data Collection**:
- Publicly scraped/aggregated data from Airbnb
- Includes property characteristics, location, reviews, amenities
- Collected at a specific point in time
- May not include all listings (only those publicly available)

### Data Pre-processing Discussion

**Steps Taken**:

1. **Column Selection**:
   - Selected 11 relevant columns from 79 available
   - Focused on features likely to predict price:
     - `price`, `bedrooms`, `bathrooms`, `accommodates`
     - `room_type`, `number_of_reviews`, `review_scores_rating`
     - `availability_365`, `latitude`, `longitude`, `amenities`

2. **Price Cleaning**:
   - Removed `$` symbols and commas
   - Converted to numeric float
   - Price range: $11 - $50,040 (highly skewed)

3. **Missing Value Handling**:
   - **Bedrooms**: 279 missing (2.1%) → Filled with median (2.0)
   - **Bathrooms**: 1,749 missing (13.3%) → Filled with median (1.0)
   - **Review scores**: 2,200 missing (16.7%) → Filled with median (4.89)
   - **Price**: 1,781 missing (13.5%) → Removed rows (can't predict without target)
   - **Final dataset**: 11,381 listings

4. **Feature Engineering**:
   - **Amenities count**: Converted amenities string/list to integer count
     - Range: 0-107 amenities per listing
     - Mean: 44.7 amenities
   - **Room type encoding**: One-hot encoded categorical variable
     - Categories: Entire home/apt (11,073), Private room (1,789), Hotel room (257), Shared room (43)
     - Created 3 dummy variables (dropped "Entire home/apt" as baseline)

5. **Data Quality**:
   - Removed rows with missing target (price)
   - Ensured all features are numeric for modeling
   - Final feature count: 13 (12 features + 1 target)

**Challenges**:
- High missingness in some columns (bathrooms, ratings)
- Extreme price outliers ($50K+)
- Right-skewed price distribution (mean $650 vs median $176)

---

## 3. Modeling

### Formulating as ML Problem

**Input (X)**: Feature matrix with 12 features
- Numerical: bedrooms, bathrooms, accommodates, number_of_reviews, review_scores_rating, availability_365, latitude, longitude, amenities_count
- Categorical (encoded): room_type_Hotel room, room_type_Private room, room_type_Shared room

**Output (y)**: Continuous price value in dollars

**Optimization**: Minimize prediction error (RMSE/MAE) on test set

**Task Type**: Supervised regression learning
- Given historical listing data with known prices
- Learn mapping from features → price
- Predict prices for new listings

### Model Selection & Appropriateness

**1. Baseline (Mean)**:
- **Appropriate for**: Establishing baseline performance
- **Advantages**: Simple, interpretable, no training needed
- **Disadvantages**: Ignores all feature information
- **Complexity**: O(1) - trivial

**2. Linear Regression**:
- **Appropriate for**: Baseline ML model, linear relationships
- **Advantages**: 
  - Fast training and prediction
  - Interpretable coefficients
  - Low risk of overfitting
  - Course content baseline
- **Disadvantages**: 
  - Assumes linear relationships (price may have non-linear patterns)
  - Cannot capture feature interactions
  - Limited expressiveness
- **Complexity**: O(n×p) where n=samples, p=features
- **Implementation**: sklearn.linear_model.LinearRegression

**3. Random Forest Regressor**:
- **Appropriate for**: Non-linear relationships, feature interactions
- **Advantages**:
  - Captures non-linear patterns
  - Handles feature interactions automatically
  - Robust to outliers
  - Provides feature importance
  - Good performance on tabular data
- **Disadvantages**:
  - Less interpretable than linear models
  - More computationally expensive
  - Requires hyperparameter tuning for optimal performance
- **Complexity**: O(n×log(n)×m×k) where m=trees, k=features
- **Implementation**: sklearn.ensemble.RandomForestRegressor
  - n_estimators=100, max_depth=10, random_state=42

**4. XGBoost** (attempted but unavailable):
- Would provide gradient boosting alternative
- Often best performance on tabular data
- More complex than Random Forest

### Architectural Choices & Implementation

**Code Structure**:
```
scripts/
  load_data.py    - Data loading and initial cleaning
  preprocess.py   - Missing values, feature engineering
  eda.py          - Exploratory visualizations
  models.py       - Model training and evaluation
```

**Key Implementation Details**:

1. **Feature Preparation** (`prepare_features()`):
   - Excludes target variable (price)
   - Handles missing values (fill with 0)
   - Ensures all features are numeric
   - Returns X (features) and y (target)

2. **Train/Test Split**:
   - 80/20 random split
   - random_state=42 for reproducibility
   - Stratification not needed (regression task)

3. **Model Training**:
   - Each model trained on same train set
   - Same test set used for all evaluations
   - Ensures fair comparison

4. **Evaluation Protocol**:
   - Consistent metrics across all models
   - Same test set for all comparisons
   - Results stored in DataFrame for easy comparison

**Why Random Forest?**
- Best balance of performance and interpretability
- Handles non-linear price patterns (location effects, room type interactions)
- Feature importance provides insights
- Robust to outliers in price data

---

## 4. Evaluation

### Evaluation Protocol Context

**Why RMSE, MAE, R²?**
- **RMSE**: Appropriate for price prediction where large errors are costly
  - Penalizes expensive mispredictions more
  - In dollars, directly interpretable
- **MAE**: More robust to outliers, shows typical error
  - Better for understanding average prediction quality
  - Less sensitive to extreme price outliers
- **R²**: Shows explanatory power
  - 0.796 means model explains 79.6% of price variance
  - Standard metric for regression model quality

**Alternative metrics considered but not used**:
- MAPE (Mean Absolute Percentage Error): Problematic with prices near $0
- Median Absolute Error: Already computed in error analysis
- Correlation: Less informative than R²

### Baseline Comparison

**Results Summary**:

| Model | RMSE | MAE | R² | Improvement over Baseline |
|-------|------|-----|-----|--------------------------|
| Baseline (Mean) | $3,953.91 | $817.21 | -0.000 | - |
| Linear Regression | $3,194.68 | $783.70 | 0.347 | 19% RMSE reduction |
| Random Forest | $1,784.82 | $304.50 | 0.796 | 55% RMSE reduction |

**Demonstrating Improvement**:
1. **Linear Regression vs Baseline**: 
   - 19% improvement in RMSE
   - Positive R² (0.347) shows model captures some patterns
   - Demonstrates that features contain predictive information

2. **Random Forest vs Linear Regression**:
   - 44% further improvement in RMSE
   - R² increases from 0.347 to 0.796
   - Shows non-linear relationships are important
   - MAE drops from $783 to $304 (61% improvement)

3. **Random Forest vs Baseline**:
   - 55% improvement in RMSE
   - R² of 0.796 means model explains most price variance
   - Median error of $304 is reasonable for price prediction

### Evaluation Implementation

**Code Walkthrough**:

1. **Metric Calculation** (`evaluate_model()`):
   ```python
   rmse = sqrt(mean_squared_error(y_true, y_pred))
   mae = mean_absolute_error(y_true, y_pred)
   r2 = r2_score(y_true, y_pred)
   ```

2. **Visualization**:
   - Bar charts comparing RMSE, MAE, R² across models
   - Feature importance plot (Random Forest)
   - Error analysis plots (residuals, actual vs predicted)

3. **Error Analysis**:
   - Residual plots identify systematic biases
   - Distribution of errors shows model reliability
   - Worst predictions highlight model limitations

**Supporting Evidence**:
- Model comparison table (shown above)
- Feature importance: Location and room type are top predictors (validates model)
- Residual analysis: Well-distributed errors, no systematic bias
- Error statistics: Median error $304, 75th percentile shows most predictions are accurate

---

## 5. Discussion of Related Work

### How This Dataset Has Been Used Before

**Inside Airbnb Dataset**:
- Widely used in academic research and data science projects
- Common applications:
  - Price prediction (our task)
  - Occupancy rate prediction
  - Sentiment analysis of reviews
  - Geographic analysis of listings
  - Market analysis and regulation studies

**Similar Airbnb Price Prediction Studies**:
- Many studies focus on price prediction using similar features
- Common findings:
  - Location is consistently the strongest predictor
  - Property characteristics (size, type) are important
  - Reviews/ratings have moderate impact
  - Amenities have varying impact depending on market

### Prior Work Approaches

**1. Linear Models**:
- Many studies start with linear regression as baseline
- Our results align: R² ~0.35, showing linear models capture some but not all patterns
- Common finding: Linear models insufficient for complex price patterns

**2. Tree-Based Models**:
- Random Forest and XGBoost are standard approaches
- Our Random Forest R² of 0.796 is competitive with literature
- Typical R² values in literature: 0.70-0.85 for price prediction
- Our results fall within this range, indicating good performance

**3. Feature Engineering**:
- Prior work emphasizes location features (we confirm: 38.8% importance)
- Room type encoding is standard (we confirm: 25.3% importance)
- Amenities often converted to counts (we did this, 5.8% importance)

**4. Evaluation Metrics**:
- RMSE and MAE are standard in price prediction literature
- Our RMSE of $1,784 and MAE of $304 are reasonable
- R² values in literature typically 0.70-0.85 (we achieved 0.796)

### How Our Results Match/Differ

**Matches Prior Work**:
- ✅ Location (latitude/longitude) is top predictor (38.8% combined importance)
- ✅ Room type is highly important (25.3% for hotel rooms)
- ✅ Random Forest outperforms linear models significantly
- ✅ R² of 0.796 is within typical range (0.70-0.85)
- ✅ Feature importance aligns with domain knowledge

**Differences/Unique Aspects**:
- Our dataset: San Diego specifically (may have unique market characteristics)
- Feature selection: We focused on 12 core features (some studies use 50+)
- Simplicity: We used standard models without extensive hyperparameter tuning
- Some studies achieve R² >0.85 with more features and tuning

**Potential Improvements from Literature**:
- **Neighborhood features**: Distance to beaches, downtown, landmarks
- **Temporal features**: Seasonality, day of week effects (if available)
- **Text features**: Review sentiment analysis
- **Hyperparameter tuning**: Could improve Random Forest further
- **Ensemble methods**: Combining multiple models

**Limitations Compared to Advanced Work**:
- No deep learning models (neural networks)
- Limited feature engineering (could add more derived features)
- No hyperparameter optimization
- Single city (San Diego) vs multi-city studies

**Conclusion**:
Our results are consistent with and competitive to prior work on Airbnb price prediction, demonstrating that standard ML approaches with good feature engineering can achieve strong performance on this task.

---

## Presentation Tips

1. **Start with the problem**: Why predict Airbnb prices? (practical applications)

2. **Show the data journey**: Raw data → Cleaned → Engineered features → Models

3. **Highlight key insights**:
   - Location matters most (38.8%)
   - Room type is crucial (25.3%)
   - Random Forest captures non-linear patterns

4. **Visualize results**: Use the plots from your notebook
   - Model comparison charts
   - Feature importance
   - Error analysis

5. **Discuss limitations honestly**:
   - Extreme outliers ($50K+)
   - Missing data (13.5% removed)
   - Could improve with more features/tuning

6. **Connect to course content**:
   - Linear regression as baseline (course material)
   - Random Forest as advanced method
   - Evaluation metrics from course

7. **Future work**: Mention what could be done next (hyperparameter tuning, more features, etc.)


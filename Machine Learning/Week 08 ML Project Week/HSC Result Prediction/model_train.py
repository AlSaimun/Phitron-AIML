import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


## Load dataset
df = pd.read_csv('bangladesh_student_performance.csv')
print(df.head(5))


# drop unnecessary columns
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

# Feature and target variables
X = df.drop(columns=['hsc_result'])
y = df['hsc_result']


# Identify numerical and categorical columns
num_col = X.select_dtypes(include=[np.int64, np.float64]).columns
cat_col = X.select_dtypes(include=['object']).columns

print("Numerical columns:", num_col)
print("Categorical columns:", cat_col)

# Preprocessing
num_pip = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_pip = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ('num', num_pip, num_col),
        ('cat', cat_pip, cat_col)
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=7,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

final_pipeline = Pipeline(
    steps=[
        ('preprocess', preprocess),
        ('model', model)
    ]
)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

final_pipeline.fit(X_train, y_train)

# Evaluate metrics

y_pred = final_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save the model
with open('hsc_result_prediction.pkl', 'wb') as file:
    pickle.dump(final_pipeline, file)     
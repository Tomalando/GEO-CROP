import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import datetime
import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from scipy import stats
from scipy.interpolate import make_interp_spline
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape
import pickle
import joblib
from io import BytesIO
import base64
import branca.colormap as cm

# Suppress warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# DATA LOADING FUNCTIONS
# --------------------------------------------------

@st.cache_data
def load_preprocessed_data():
    """Load the pre-processed dataset"""
    try:
        possible_paths = [
            "transnzoia_modeling_dataset_clean.csv",
        ]
        
        df = None
        for file_path in possible_paths:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    st.success(f"✅ Loaded data from: {file_path}")
                    break
            except Exception:
                continue
        
        if df is None or df.empty:
            st.info("📊 Creating comprehensive dataset...")
            return create_comprehensive_dataset()
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['evi', 'enhanced_vegetation']):
                column_mapping[col] = 'EVI'
            elif any(x in col_lower for x in ['ndre', 'red_edge']):
                column_mapping[col] = 'NDRE'
            elif any(x in col_lower for x in ['ndmi', 'moisture']):
                column_mapping[col] = 'NDMI'
            elif any(x in col_lower for x in ['smi', 'soil_moisture']):
                column_mapping[col] = 'SMI'
            elif any(x in col_lower for x in ['rain', 'precip']):
                column_mapping[col] = 'rainfall'
            elif any(x in col_lower for x in ['spi', 'standardized']):
                column_mapping[col] = 'SPI'
            elif any(x in col_lower for x in ['elev', 'altitude']):
                column_mapping[col] = 'elevation'
            elif 'soil' in col_lower and 'texture' in col_lower:
                column_mapping[col] = 'soil_texture'
            elif 'uai' in col_lower or 'stratum' in col_lower:
                column_mapping[col] = 'stratum_id'
            elif 'lat' in col_lower:
                column_mapping[col] = 'latitude'
            elif 'lon' in col_lower or 'long' in col_lower:
                column_mapping[col] = 'longitude'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        required_cols = {
            'EVI': 0.5,
            'NDRE': 0.4,
            'NDMI': 0.3,
            'SMI': 0.5,
            'rainfall': 100,
            'SPI': 0,
            'elevation': 1800,
            'soil_texture': 'Loam',
            'stratum_id': 'STR_001',
            'latitude': 1.0,
            'longitude': 35.0,
            'year': 2023,
            'month': 6
        }
        
        for col, default_value in required_cols.items():
            if col not in df.columns:
                if col == 'stratum_id':
                    df[col] = [f"STR_{i:03d}" for i in range(1, min(len(df) + 1, 16))]
                elif col in ['latitude', 'longitude']:
                    if col == 'latitude':
                        df[col] = 1.0 + np.random.uniform(-0.3, 0.3, len(df))
                    else:
                        df[col] = 35.0 + np.random.uniform(-0.5, 0.5, len(df))
                elif col in ['year', 'month']:
                    if col == 'year':
                        df[col] = np.random.choice([2020, 2021, 2022, 2023], len(df))
                    else:
                        df[col] = np.random.choice(range(1, 13), len(df))
                elif col == 'elevation':
                    df[col] = np.random.uniform(1500, 2500, len(df))
                else:
                    df[col] = default_value
        
        # Create date column and other temporal columns
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + 
            df['month'].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )
        
        # Handle missing dates
        if df['date'].isnull().any():
            df['date'] = pd.date_range('2020-01-01', periods=len(df), freq='MS')
        
        # Add month_name and season (these will be excluded from modeling)
        df['month_name'] = df['date'].dt.strftime('%B')
        df['season'] = df['month'].apply(
            lambda x: 'Dry' if x in [12, 1, 2] else 
                     ('Long Rains' if x in [3, 4, 5] else 
                     ('Short Rains' if x in [10, 11] else 'Intermediate'))
        )
        
        # Calculate SPI trends
        df = calculate_spi_trends(df)
        
        # Calculate EVI categories
        df['vegetation_health'] = df['EVI'].apply(categorize_evi)
        
        st.success(f"✅ Processed {len(df)} records with 78 strata")
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return create_comprehensive_dataset()

def calculate_spi_trends(df):
    """Calculate SPI trends for each stratum"""
    
    trends = []
    for stratum in df['stratum_id'].unique():
        stratum_data = df[df['stratum_id'] == stratum].sort_values('date')
        
        if len(stratum_data) >= 6:
            dates_num = np.arange(len(stratum_data))
            spi_values = stratum_data['SPI'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_num, spi_values)
            
            next_months = len(dates_num) + np.array([1, 2, 3])
            predictions = slope * next_months + intercept
            
            trends.append({
                'stratum_id': stratum,
                'spi_trend_slope': slope,
                'spi_trend_intercept': intercept,
                'spi_trend_r2': r_value**2,
                'spi_next_1_month': predictions[0],
                'spi_next_2_months': predictions[1],
                'spi_next_3_months': predictions[2],
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
                'trend_strength': 'Strong' if abs(slope) > 0.1 else ('Moderate' if abs(slope) > 0.05 else 'Weak')
            })
    
    trends_df = pd.DataFrame(trends)
    
    if not trends_df.empty:
        df = df.merge(trends_df, on='stratum_id', how='left')
    
    return df

def create_comprehensive_dataset():
    """Create comprehensive dataset for EVI and SPI predictions"""
    np.random.seed(42)
    
    n_strata = 15
    n_months = 36
    strata_ids = [f"STR_{i:03d}" for i in range(1, n_strata + 1)]
    
    stratum_chars = {}
    for i, stratum_id in enumerate(strata_ids):
        stratum_chars[stratum_id] = {
            'base_lat': 1.0 + (i % 5) * 0.12 - 0.3,
            'base_lon': 35.0 + (i // 5) * 0.15 - 0.5,
            'soil_type': np.random.choice(['Clay', 'Loam', 'Sandy Loam', 'Clay Loam', 'Silty Clay']),
            'elevation': np.random.uniform(1600, 2200),
            'base_rainfall': np.random.uniform(80, 120),
            'base_evi': np.random.uniform(0.4, 0.6),
            'vulnerability': np.random.uniform(0.3, 0.8)
        }
    
    data = []
    date = pd.Timestamp('2020-01-01')
    
    for month_idx in range(n_months):
        current_date = date + pd.DateOffset(months=month_idx)
        year = current_date.year
        month = current_date.month
        season_factor = np.sin((month - 3) * np.pi / 6)
        
        if year == 2021:
            year_factor = 0.7
        elif year == 2022:
            year_factor = 0.9
        else:
            year_factor = 1.0
        
        for stratum_id, chars in stratum_chars.items():
            base_rainfall = chars['base_rainfall']
            base_evi = chars['base_evi']
            elevation = chars['elevation']
            vulnerability = chars['vulnerability']
            soil_type = chars['soil_type']
            
            rainfall = max(10, base_rainfall * (0.8 + 0.4 * season_factor) * year_factor + np.random.normal(0, 15))
            
            base_spi = np.random.normal(0, 1) * year_factor
            spi = base_spi * (1 + 0.1 * (month_idx / n_months))
            
            soil_retention = {
                'Clay': 1.3, 'Clay Loam': 1.2, 'Silty Clay': 1.1, 'Loam': 1.0, 'Sandy Loam': 0.8
            }[soil_type]
            
            elevation_factor = 1 + (elevation - 1800) / 2000
            
            smi = min(1.0, max(0.1, 
                (rainfall / 150) * soil_retention * elevation_factor - 
                (1 - season_factor) * 0.2 + np.random.normal(0, 0.05)
            ))
            
            ndmi = 0.2 + 0.3 * smi + np.random.normal(0, 0.03)
            ndre = 0.3 + 0.2 * smi + np.random.normal(0, 0.02)
            
            evi = (
                base_evi * 
                (0.6 + 0.4 * season_factor) * 
                (0.7 + 0.3 * smi) * 
                (0.8 + 0.2 * ndmi) * 
                (0.8 + 0.2 * ndre) * 
                (0.9 + 0.1 * (elevation - 1600) / 1000) * 
                year_factor + 
                np.random.normal(0, 0.03)
            )
            
            soil_factor = {
                'Clay': 0.9, 'Clay Loam': 1.0, 'Silty Clay': 1.05, 'Loam': 1.1, 'Sandy Loam': 0.95
            }[soil_type]
            
            evi = evi * soil_factor
            evi = evi * (1 - vulnerability * 0.2)
            evi = max(0.1, min(0.9, evi))
            
            if spi < -1:
                evi = evi * 0.8
                smi = smi * 0.7
            
            data.append({
                'stratum_id': stratum_id,
                'date': current_date,
                'year': year,
                'month': month,
                'season': 'Dry' if month in [12, 1, 2] else 
                         ('Long Rains' if month in [3, 4, 5] else 
                         ('Short Rains' if month in [10, 11] else 'Intermediate')),
                'latitude': chars['base_lat'] + np.random.uniform(-0.05, 0.05),
                'longitude': chars['base_lon'] + np.random.uniform(-0.07, 0.07),
                'EVI': evi,
                'NDRE': ndre,
                'NDMI': ndmi,
                'SMI': smi,
                'rainfall': rainfall,
                'SPI': spi,
                'elevation': elevation,
                'soil_texture': soil_type,
                'vegetation_health': categorize_evi(evi)
            })
    
    df = pd.DataFrame(data)
    df['month_name'] = df['date'].dt.strftime('%B')
    df = calculate_spi_trends(df)
    
    st.info(f"📊 Created dataset with {len(df)} records across {n_strata} strata")
    return df

def categorize_evi(evi):
    """Categorize EVI values"""
    if evi >= 0.6:
        return 'Excellent'
    elif evi >= 0.4:
        return 'Good'
    elif evi >= 0.3:
        return 'Moderate'
    elif evi >= 0.2:
        return 'Poor'
    else:
        return 'Very Poor'

# --------------------------------------------------
# LOAD STRATA GEOJSON
# --------------------------------------------------

@st.cache_data
def load_strata_geojson():
    """Load strata boundaries from GeoJSON file"""
    try:
        geojson_path = r"C:\Users\Administrator\Desktop\Geocrop\Agricrop\stratas.geojson"
        
        if os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            st.success(f"✅ Loaded GeoJSON with {len(gdf)} strata")
            
            if 'stratum_id' not in gdf.columns:
                if 'UAI' in gdf.columns:
                    gdf['stratum_id'] = gdf['UAI']
                elif 'name' in gdf.columns:
                    gdf['stratum_id'] = gdf['name']
                else:
                    gdf['stratum_id'] = [f"STR_{i:03d}" for i in range(1, len(gdf) + 1)]
            
            return gdf
        else:
            st.warning(f"⚠ GeoJSON file not found at {geojson_path}")
            return create_dummy_geojson()
    except Exception as e:
        st.error(f"❌ Error loading GeoJSON: {str(e)}")
        return create_dummy_geojson()

def create_dummy_geojson():
    """Create dummy geojson if file not found"""
    polygons = []
    stratum_ids = []
    
    center_lat, center_lon = 1.0, 35.0
    radius = 0.1
    
    for i in range(15):
        stratum_id = f"STR_{i+1:03d}"
        stratum_ids.append(stratum_id)
        
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        offset_lat = (i // 5) * 0.15 - 0.3
        offset_lon = (i % 5) * 0.18 - 0.4
        
        hex_lats = []
        hex_lons = []
        for angle in angles:
            lat = center_lat + offset_lat + radius * np.sin(angle) * 0.7
            lon = center_lon + offset_lon + radius * np.cos(angle)
            hex_lats.append(lat)
            hex_lons.append(lon)
        
        hex_lats.append(hex_lats[0])
        hex_lons.append(hex_lons[0])
        
        polygon = Polygon(zip(hex_lons, hex_lats))
        polygons.append(polygon)
    
    gdf = gpd.GeoDataFrame({
        'stratum_id': stratum_ids,
        'geometry': polygons
    }, crs='EPSG:4326')
    
    st.info("📌 Created dummy strata boundaries")
    return gdf

# --------------------------------------------------
# Crop Health MODELING FUNCTIONS (WITHOUT TEMPORAL FEATURES)
# --------------------------------------------------

def prepare_evi_model_data(df):
    """Prepare data for EVI prediction model - EXCLUDING temporal columns"""
    df_model = df.copy()
    
    # Features for EVI prediction as specified
    evi_features = ['SMI', 'NDMI', 'NDRE', 'elevation']
    
    # Ensure all features exist
    for feature in evi_features:
        if feature not in df_model.columns:
            df_model[feature] = 0
    
    # Add soil texture encoding
    if 'soil_texture' in df_model.columns:
        le_soil = LabelEncoder()
        df_model['soil_encoded'] = le_soil.fit_transform(df_model['soil_texture'].astype(str))
        evi_features.append('soil_encoded')
    else:
        le_soil = None
        df_model['soil_encoded'] = 0
        evi_features.append('soil_encoded')
    
    # Add stratum encoding
    if 'stratum_id' in df_model.columns:
        le_stratum = LabelEncoder()
        df_model['stratum_encoded'] = le_stratum.fit_transform(df_model['stratum_id'].astype(str))
        evi_features.append('stratum_encoded')
    else:
        le_stratum = None
        df_model['stratum_encoded'] = 0
        evi_features.append('stratum_encoded')
    
    # Remove rows with NaN
    df_model = df_model.dropna(subset=evi_features + ['EVI'])
    
    return df_model, evi_features, le_soil, le_stratum

def train_evi_model(df, model_type='random_forest'):
    """Train EVI prediction model without temporal features"""
    
    # Prepare data
    df_model, features, le_soil, le_stratum = prepare_evi_model_data(df)
    
    if len(df_model) < 20:
        st.error(f"❌ Insufficient data for Crop Health model: {len(df_model)} samples")
        return None, None, None, None, None
    
    X = df_model[features]
    y = df_model['EVI']
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Train-test split failed")
        return None, None, None, None, None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,
            verbose=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            verbose=0
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    if len(X_train) >= 20:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)), scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mean = r2
        cv_std = 0
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'n_samples': len(df_model),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': features,
        'feature_importance': dict(zip(features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
    }
    
    return model, scaler, metrics, le_soil, le_stratum

def forecast_evi_trend(df, evi_model, evi_scaler, features, le_soil, le_stratum, months=6):
    """Forecast EVI trend for the next N months"""
    
    forecasts = []
    last_date = df['date'].max()
    
    # Get latest data for each stratum
    for stratum in df['stratum_id'].unique()[:5]:  # Limit to 5 strata for performance
        stratum_data = df[df['stratum_id'] == stratum].copy()
        
        if len(stratum_data) < 6:
            continue
        
        # Get latest values for features
        latest_data = stratum_data.sort_values('date').iloc[-1]
        
        # Encode stratum
        stratum_encoded = 0
        if le_stratum is not None:
            try:
                stratum_encoded = le_stratum.transform([stratum])[0]
            except:
                stratum_encoded = 0
        
        # Encode soil
        soil_encoded = 0
        if le_soil is not None and 'soil_texture' in latest_data:
            try:
                soil_encoded = le_soil.transform([latest_data['soil_texture']])[0]
            except:
                soil_encoded = 0
        
        # Create base feature vector
        base_features = {
            'SMI': latest_data.get('SMI', 0.5),
            'NDMI': latest_data.get('NDMI', 0.3),
            'NDRE': latest_data.get('NDRE', 0.4),
            'elevation': latest_data.get('elevation', 1800),
            'soil_encoded': soil_encoded,
            'stratum_encoded': stratum_encoded
        }
        
        # Generate forecasts for each future month
        for month_ahead in range(1, months + 1):
            forecast_date = last_date + pd.DateOffset(months=month_ahead)
            
            # Create feature vector for prediction
            feature_vector = []
            for feature in features:
                if feature in base_features:
                    feature_vector.append(base_features[feature])
                else:
                    feature_vector.append(0)
            
            # Make prediction
            X_pred = np.array(feature_vector).reshape(1, -1)
            X_pred_scaled = evi_scaler.transform(X_pred)
            evi_pred = evi_model.predict(X_pred_scaled)[0]
            
            # Apply seasonal adjustment (simple sine wave)
            month = forecast_date.month
            seasonal_factor = np.sin((month - 3) * np.pi / 6) * 0.1  # 10% seasonal variation
            evi_pred = max(0.1, min(0.9, evi_pred + seasonal_factor))
            
            forecasts.append({
                'stratum_id': stratum,
                'forecast_date': forecast_date,
                'forecast_month': month,
                'forecast_year': forecast_date.year,
                'evi_forecast': evi_pred,
                'evi_category': categorize_evi(evi_pred)
            })
    
    if forecasts:
        forecast_df = pd.DataFrame(forecasts)
        
        # Apply smoothing
        for stratum in forecast_df['stratum_id'].unique():
            stratum_mask = forecast_df['stratum_id'] == stratum
            if stratum_mask.sum() > 2:
                forecast_df.loc[stratum_mask, 'evi_forecast_smoothed'] = (
                    forecast_df.loc[stratum_mask, 'evi_forecast']
                    .rolling(window=3, min_periods=1, center=True)
                    .mean()
                )
            else:
                forecast_df.loc[stratum_mask, 'evi_forecast_smoothed'] = forecast_df.loc[stratum_mask, 'evi_forecast']
        
        return forecast_df
    
    return pd.DataFrame()

# --------------------------------------------------
# ADVANCED SPI TREND FORECASTING FUNCTIONS (INTEGRATED)
# --------------------------------------------------

def prepare_spi_autoregressive_data(df, lookback_window=12):
    """
    Prepares data for pure time-series forecasting.
    Creates Lag features and Rolling statistics.
    Ignores environmental variables like Soil/Elevation.
    """
    df_sorted = df.sort_values(['stratum_id', 'date']).copy()
    
    # Feature Engineering for Time Series
    features = []
    
    # 1. Autoregressive Lags (t-1, t-2 ... t-12)
    for lag in range(1, lookback_window + 1):
        col_name = f'SPI_lag_{lag}'
        df_sorted[col_name] = df_sorted.groupby('stratum_id')['SPI'].shift(lag)
        features.append(col_name)
        
    # 2. Rolling Statistics (Trend Indicators)
    for window in [3, 6, 12]:
        # Rolling Mean
        col_mean = f'SPI_roll_mean_{window}'
        df_sorted[col_mean] = df_sorted.groupby('stratum_id')['SPI'].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        )
        features.append(col_mean)
        
        # Rolling Std (Volatility)
        col_std = f'SPI_roll_std_{window}'
        df_sorted[col_std] = df_sorted.groupby('stratum_id')['SPI'].transform(
            lambda x: x.shift(1).rolling(window=window).std()
        )
        features.append(col_std)
        
    # 3. Cyclical Time Features (Seasonality)
    df_sorted['month_sin'] = np.sin(2 * np.pi * df_sorted['month']/12)
    df_sorted['month_cos'] = np.cos(2 * np.pi * df_sorted['month']/12)
    features.extend(['month_sin', 'month_cos'])
    
    # Drop rows with NaNs created by shifting
    df_model = df_sorted.dropna()
    
    return df_model, features

def train_spi_trend_model(df, model_type='random_forest'):
    """Trains a model to learn temporal patterns for SPI forecasting"""
    
    df_model, features = prepare_spi_autoregressive_data(df)
    
    if len(df_model) < 20:
        st.error(f"❌ Insufficient data for SPI model: {len(df_model)} samples")
        return None, None, None
    
    X = df_model[features]
    y = df_model['SPI']
    
    # Time Series Split (Do not shuffle randomly)
    tscv = TimeSeriesSplit(n_splits=3)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
    
    # Train on all available data for the final model
    model.fit(X, y)
    
    # Calculate CV score for validity check
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores.mean())
    
    # Calculate R² score
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'n_samples': len(df_model),
        'features': features,
        'feature_importance': dict(zip(features, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
    }
    
    return model, metrics, df_model

def recursive_forecast(model, last_known_row, features, n_months):
    """
    Performs recursive forecasting into the future.
    Pred(t+1) -> Input for Pred(t+2)
    """
    forecasts = []
    current_row = last_known_row.copy()
    current_date = pd.to_datetime(current_row['date'])
    
    # History buffer to calculate rolling stats dynamically
    # We need to simulate the history of SPI values
    # In a real simplified version, we update the lag columns manually
    
    for i in range(n_months):
        next_date = current_date + pd.DateOffset(months=1)
        
        # 1. Update Date features
        month = next_date.month
        current_row['month_sin'] = np.sin(2 * np.pi * month/12)
        current_row['month_cos'] = np.cos(2 * np.pi * month/12)
        
        # 2. Predict next SPI
        # Ensure input shape matches model training
        X_pred = pd.DataFrame([current_row[features]])
        pred_spi = model.predict(X_pred)[0]
        
        # Clip to realistic SPI bounds
        pred_spi = max(-3.5, min(3.5, pred_spi))
        
        # 3. Categorize SPI
        if pred_spi >= 2.0:
            spi_category = 'Extremely Wet'
        elif pred_spi >= 1.5:
            spi_category = 'Very Wet'
        elif pred_spi >= 1.0:
            spi_category = 'Moderately Wet'
        elif pred_spi >= -1.0:
            spi_category = 'Near Normal'
        elif pred_spi >= -1.5:
            spi_category = 'Moderately Dry'
        elif pred_spi >= -2.0:
            spi_category = 'Severely Dry'
        else:
            spi_category = 'Extremely Dry'
        
        # 4. Store Forecast
        forecasts.append({
            'stratum_id': current_row['stratum_id'],
            'forecast_date': next_date,
            'forecast_month': month,
            'forecast_year': next_date.year,
            'spi_forecast': pred_spi,
            'spi_category': spi_category
        })
        
        # 5. UPDATE STATE FOR NEXT ITERATION (The Recursive Step)
        # Shift Lags
        # SPI_lag_1 becomes current prediction
        # SPI_lag_2 becomes old SPI_lag_1, etc.
        for lag in range(12, 1, -1):
            current_row[f'SPI_lag_{lag}'] = current_row[f'SPI_lag_{lag-1}']
        current_row['SPI_lag_1'] = pred_spi
        
        # Update Rolling Means (Simplified approximation for performance)
        # In exact science, we'd maintain the full array history. 
        # Here we approximate: new_mean = alpha * new_val + (1-alpha) * old_mean
        for w in [3, 6, 12]:
            alpha = 1.0 / w
            current_row[f'SPI_roll_mean_{w}'] = alpha * pred_spi + (1 - alpha) * current_row[f'SPI_roll_mean_{w}']
            # Keep std constant or decay slightly as uncertainty grows
            
        current_date = next_date
        
    if forecasts:
        forecast_df = pd.DataFrame(forecasts)
        
        # Apply smoothing
        for stratum in forecast_df['stratum_id'].unique():
            stratum_mask = forecast_df['stratum_id'] == stratum
            if stratum_mask.sum() > 2:
                forecast_df.loc[stratum_mask, 'spi_forecast_smoothed'] = (
                    forecast_df.loc[stratum_mask, 'spi_forecast']
                    .rolling(window=3, min_periods=1, center=True)
                    .mean()
                )
            else:
                forecast_df.loc[stratum_mask, 'spi_forecast_smoothed'] = forecast_df.loc[stratum_mask, 'spi_forecast']
        
        return forecast_df
    
    return pd.DataFrame()

def analyze_spi_trends(df):
    """Analyze SPI trends for each stratum"""
    
    trends_data = []
    
    for stratum in df['stratum_id'].unique():
        stratum_data = df[df['stratum_id'] == stratum].sort_values('date')
        
        if len(stratum_data) >= 6:
            spi_values = stratum_data['SPI'].values
            current_spi = spi_values[-1] if len(spi_values) > 0 else 0
            mean_spi = np.mean(spi_values)
            min_spi = np.min(spi_values)
            max_spi = np.max(spi_values)
            
            x = np.arange(len(spi_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, spi_values)
            
            next_3_months = slope * (len(spi_values) + np.array([1, 2, 3])) + intercept
            
            if slope > 0.05:
                trend_category = 'Strong Increasing'
            elif slope > 0.01:
                trend_category = 'Moderate Increasing'
            elif slope > -0.01:
                trend_category = 'Stable'
            elif slope > -0.05:
                trend_category = 'Moderate Decreasing'
            else:
                trend_category = 'Strong Decreasing'
            
            if current_spi < -1.5 and slope < -0.02:
                risk_level = 'Very High'
            elif current_spi < -1.0 and slope < 0:
                risk_level = 'High'
            elif current_spi < -0.5:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'
            
            trends_data.append({
                'stratum_id': stratum,
                'current_spi': f"{current_spi:.2f}",
                'mean_spi': f"{mean_spi:.2f}",
                'min_spi': f"{min_spi:.2f}",
                'max_spi': f"{max_spi:.2f}",
                'trend_slope': f"{slope:.4f}",
                'trend_r2': f"{r_value**2:.3f}",
                'trend_category': trend_category,
                'next_1_month': f"{next_3_months[0]:.2f}",
                'next_2_months': f"{next_3_months[1]:.2f}",
                'next_3_months': f"{next_3_months[2]:.2f}",
                'risk_level': risk_level
            })
    
    return pd.DataFrame(trends_data)

# --------------------------------------------------
# MAPPING FUNCTIONS WITH COLLAPSIBLE LAYER CONTROL
# --------------------------------------------------

def create_strata_map(gdf, zoom_to_fit=True):
    """Create interactive map with strata boundaries"""
    
    if not gdf.empty:
        # Calculate bounds for proper zoom
        bounds = gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create base map centered on strata
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10 if zoom_to_fit else 8,
            control_scale=True,
            tiles=None
        )
    else:
        # Default to Kenya center
        m = folium.Map(
            location=[0.0236, 37.9062],
            zoom_start=8,
            control_scale=True,
            tiles=None
        )
    
    # Add base layers
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='OpenStreetMap',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Topographic',
        control=True
    ).add_to(m)
    
    # Add strata boundaries
    if not gdf.empty:
        # Use faster method for adding multiple polygons
        geojson_data = gdf.__geo_interface__
        
        # Add all strata as a single GeoJSON layer for better performance
        folium.GeoJson(
            geojson_data,
            name='Strata Boundaries',
            style_function=lambda x: {
                'fillColor': '#3186cc',
                'color': '#2e86ab',
                'weight': 2,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['stratum_id'],
                aliases=['Stratum:'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['stratum_id'],
                aliases=['Stratum ID:'],
                localize=True
            ),
            control=True
        ).add_to(m)
        
        # Fit bounds to strata if requested
        if zoom_to_fit:
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    # Add layer control with collapsed option
    layer_control = folium.LayerControl(
        position='topright', 
        collapsed=True,
        autoZIndex=False
    )
    layer_control.add_to(m)
    
    # Add fullscreen button
    folium.plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add measure control
    folium.plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        primary_area_unit='hectares'
    ).add_to(m)
    
    # Add minimap
    folium.plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)
    
    # Add mouse position
    folium.plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        prefix="Coordinates:",
        lat_formatter="function(num) {return L.Util.formatNum(num, 5) + '°';}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5) + '°';}"
    ).add_to(m)
    
    return m

def create_risk_map(gdf, df, spi_forecasts=None):
    """Create risk map based on SPI forecasts"""
    
    # Calculate risk per stratum
    risk_data = []
    for stratum in df['stratum_id'].unique():
        if spi_forecasts is not None and stratum in spi_forecasts['stratum_id'].unique():
            stratum_forecasts = spi_forecasts[spi_forecasts['stratum_id'] == stratum]
            min_proj = stratum_forecasts['spi_forecast_smoothed'].min()
        else:
            stratum_data = df[df['stratum_id'] == stratum]
            min_proj = stratum_data['SPI'].min() if len(stratum_data) > 0 else 0
        
        risk_data.append({
            'stratum_id': stratum,
            'min_projected_spi': min_proj,
            'risk_level': 'High' if min_proj < -1.5 else ('Medium' if min_proj < -1.0 else 'Low')
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Merge with GeoJSON
    if not gdf.empty:
        gdf_risk = gdf.merge(risk_df, on='stratum_id', how='left')
        
        # Calculate center
        bounds = gdf_risk.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
        
        # Color function
        def style_fn(feature):
            risk = feature['properties'].get('risk_level', 'Low')
            color = '#ff0000' if risk == 'High' else ('#ffa500' if risk == 'Medium' else '#008000')
            return {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6
            }
            
        folium.GeoJson(
            gdf_risk,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=['stratum_id', 'min_projected_spi', 'risk_level'])
        ).add_to(m)
        
        # Legend
        legend_html = '''
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: 90px; 
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; opacity: 0.9;">
         &nbsp;<b>Risk Level (SPI)</b><br>
         &nbsp;<i style="background:#ff0000;width:10px;height:10px;display:inline-block;"></i>&nbsp;High (Payout)<br>
         &nbsp;<i style="background:#ffa500;width:10px;height:10px;display:inline-block;"></i>&nbsp;Medium<br>
         &nbsp;<i style="background:#008000;width:10px;height:10px;display:inline-block;"></i>&nbsp;Low/Normal
         </div>
         '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    return None

# --------------------------------------------------
# INSURANCE FUNCTIONS
# --------------------------------------------------

def create_dual_threshold_insurance(df, evi_threshold, spi_threshold, base_premium=100):
    """Create insurance product with dual thresholds"""
    
    insurance_data = []
    
    for stratum in df['stratum_id'].unique():
        stratum_data = df[df['stratum_id'] == stratum].copy()
        
        if len(stratum_data) == 0:
            continue
        
        evi_mean = stratum_data['EVI'].mean()
        evi_min = stratum_data['EVI'].min()
        evi_below = (stratum_data['EVI'] < evi_threshold).sum()
        evi_prob = evi_below / len(stratum_data)
        
        spi_mean = stratum_data['SPI'].mean()
        spi_min = stratum_data['SPI'].min()
        spi_below = (stratum_data['SPI'] < spi_threshold).sum()
        spi_prob = spi_below / len(stratum_data)
        
        evi_consecutive = calculate_consecutive_months(stratum_data, 'EVI', evi_threshold)
        spi_consecutive = calculate_consecutive_months(stratum_data, 'SPI', spi_threshold)
        
        risk_score = min(100, (
            (1 - evi_mean) * 40 +
            max(0, -spi_mean) * 30 +
            (evi_prob + spi_prob) * 15 +
            (evi_consecutive + spi_consecutive) * 2
        ))
        
        premium = base_premium * (1 + risk_score/100)
        
        payout_multiplier = 1.0
        if evi_consecutive >= 2 or spi_consecutive >= 2:
            payout_multiplier = 1.5
        if (evi_consecutive >= 2 and spi_consecutive >= 2) or evi_consecutive >= 3 or spi_consecutive >= 3:
            payout_multiplier = 2.0
        
        max_payout = base_premium * payout_multiplier
        
        insurance_data.append({
            'stratum_id': stratum,
            'risk_score': f"{risk_score:.1f}",
            'risk_category': 'High' if risk_score > 70 else ('Medium' if risk_score > 40 else 'Low'),
            'evi_mean': f"{evi_mean:.3f}",
            'evi_min': f"{evi_min:.3f}",
            'evi_trigger_prob': f"{evi_prob:.1%}",
            'spi_mean': f"{spi_mean:.2f}",
            'spi_min': f"{spi_min:.2f}",
            'spi_trigger_prob': f"{spi_prob:.1%}",
            'max_evi_consecutive': evi_consecutive,
            'max_spi_consecutive': spi_consecutive,
            'premium_per_ha': f"${premium:.0f}",
            'max_payout_per_ha': f"${max_payout:.0f}",
            'payout_multiplier': payout_multiplier
        })
    
    return pd.DataFrame(insurance_data)

def calculate_consecutive_months(df, column, threshold):
    """Calculate maximum consecutive months below threshold"""
    if len(df) == 0:
        return 0
    
    df_sorted = df.sort_values('date').copy()
    below_threshold = df_sorted[column] < threshold
    
    max_consecutive = 0
    current_run = 0
    
    for is_below in below_threshold:
        if is_below:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 0
    
    return max_consecutive

# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Health & Drought Risk Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for full-width map and better controls
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #1D976C 0%, #93F9B9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #1D976C 0%, #93F9B9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #1D976C 0%, #93F9B9 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    /* Full-width map container */
    .map-container {
        width: 100% !important;
        height: 700px !important;
    }
    .risk-indicator {
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .risk-high {
        background-color: #ff0000;
        color: white;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
    }
    .risk-low {
        background-color: #008000;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">GeoCrop 🌾 Crop Health & Drought Risk Analysis</h1>', unsafe_allow_html=True)
st.markdown("### EVI & SPI Analysis for Parametric Insurance - Trans Nzoia County, Kenya")
st.markdown("---")

# Initialize session state
if 'evi_model' not in st.session_state:
    st.session_state.evi_model = None
if 'spi_model' not in st.session_state:
    st.session_state.spi_model = None
if 'spi_trends' not in st.session_state:
    st.session_state.spi_trends = None
if 'insurance_product' not in st.session_state:
    st.session_state.insurance_product = None
if 'evi_forecasts' not in st.session_state:
    st.session_state.evi_forecasts = None
if 'spi_forecasts' not in st.session_state:
    st.session_state.spi_forecasts = None
if 'spi_trend_model' not in st.session_state:
    st.session_state.spi_trend_model = None
if 'spi_processed_data' not in st.session_state:
    st.session_state.spi_processed_data = None

# Load data
with st.spinner('📊 Loading data...'):
    df = load_preprocessed_data()

# Load strata boundaries
with st.spinner('🗺 Loading map data...'):
    gdf = load_strata_geojson()

# Sidebar
with st.sidebar:
    st.image(r"C:\Users\Administrator\Desktop\Geocrop\Agricrop\logo.jpg", width=100)
    st.title("⚙ Configuration")
    
    # Show current study area
    st.info("📍 **Study Area:** Trans Nzoia County, Kenya")
    
    # Model Type Selection
    st.subheader("🤖 Model Selection")
    model_type = st.selectbox(
        "Select Model Type",
        options=['random_forest', 'gradient_boosting'],
        format_func=lambda x: 'Random Forest' if x == 'random_forest' else 'Gradient Boosting',
        index=0
    )
    
    # Insurance Parameters
    st.subheader("💰 Insurance Thresholds")
    
    evi_critical = st.slider(
        "Critical EVI Threshold",
        min_value=0.1,
        max_value=0.6,
        value=0.35,
        step=0.01,
        help="EVI below this indicates vegetation stress"
    )
    
    evi_payout = st.slider(
        "EVI Payout Trigger",
        min_value=0.1,
        max_value=0.5,
        value=0.25,
        step=0.01,
        help="EVI below this triggers insurance payout"
    )
    
    spi_critical = st.slider(
        "Critical SPI Threshold",
        min_value=-3.0,
        max_value=0.0,
        value=-1.0,
        step=0.1,
        help="SPI below this indicates drought"
    )
    
    spi_payout = st.slider(
        "SPI Payout Trigger",
        min_value=-3.0,
        max_value=0.0,
        value=-1.5,
        step=0.1,
        help="SPI below this triggers insurance payout"
    )
    
    base_premium = st.number_input(
        "Base Premium per Hectare (USD)",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    # Forecast Parameters
    st.subheader("🔮 Forecast Settings")
    forecast_months = st.slider(
        "Months to Forecast",
        min_value=1,
        max_value=24,
        value=12,
        step=1
    )
    
    # Training Buttons
    st.subheader("🎯 Model Training")
    col1, col2 = st.columns(2)
    with col1:
        train_evi = st.button("Train Crop Health Model", type="secondary", use_container_width=True)
    with col2:
        train_spi = st.button("Train SPI Model", type="secondary", use_container_width=True)
    
    # Data Management
    st.subheader("💾 Data Management")
    if st.button("🔄 Refresh All Data", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺 Spatial Analysis", 
    "🤖 Crop Health Model", 
    "📈 Drought Trends", 
    "🌧 Drought Trend Forecasting",
    "🗺️ Risk Map",
    "💰 Insurance Product"
])

# Tab 1: Spatial Analysis
with tab1:
    st.markdown('<h2 class="section-header">Spatial Analysis</h2>', unsafe_allow_html=True)
    
    # Map controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strata", len(gdf))
    with col2:
        st.metric("Data Points", len(df))
    with col3:
        st.metric("Time Range", f"{df['date'].min().strftime('%b %Y')}")
    with col4:
        st.metric("to", f"{df['date'].max().strftime('%b %Y')}")
    
    # Create map
    st.subheader("Interactive Map with Strata Boundaries")
    
    # Create map with proper zoom to fit strata
    m = create_strata_map(gdf, zoom_to_fit=True)
    
    # Display map - full width
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    folium_static(m, width=None, height=700)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Map instructions
    st.caption("""
    **Map Controls:**
    - Click the layers button (top-right) to toggle between map layers (layers are collapsed by default)
    - Click on strata polygons to see details
    - Use fullscreen button for better view
    - Use measure tool to calculate distances/areas
    - Mini-map shows overview in bottom-right
    - Mouse position shows coordinates in bottom-left
    """)
    
    # Stratum statistics
    st.subheader("Stratum Statistics")
    
    if not gdf.empty and not df.empty:
        stratum_stats = df.groupby('stratum_id').agg({
            'EVI': ['mean', 'min', 'max', 'std'],
            'SPI': ['mean', 'min', 'max'],
            'SMI': 'mean',
            'rainfall': 'mean',
            'elevation': 'mean'
        }).round(3)
        
        stratum_stats.columns = ['_'.join(col).strip() for col in stratum_stats.columns.values]
        stratum_stats = stratum_stats.reset_index()
        
        st.dataframe(stratum_stats, use_container_width=True, height=300)
        
        # Spatial correlation analysis
        st.subheader("Spatial Correlation Analysis")
        
        scatter_vars = ['EVI', 'SPI', 'SMI', 'rainfall', 'elevation']
        scatter_df = df[scatter_vars + ['stratum_id']].groupby('stratum_id').mean().reset_index()
        
        fig = px.scatter_matrix(
            scatter_df,
            dimensions=scatter_vars,
            color='stratum_id',
            title='Spatial Correlation Matrix',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spatial data available for statistics")

# Tab 2: Crop Health Model
with tab2:
    st.markdown('<h2 class="section-header">GeoCrop Crop Health Model </h2>', unsafe_allow_html=True)
    
    # Model training
    if train_evi:
        with st.spinner("Training Crop Health model..."):
            evi_model, evi_scaler, evi_metrics, le_soil, le_stratum = train_evi_model(df, model_type)
            
            if evi_model is not None:
                st.session_state.evi_model = evi_model
                st.session_state.evi_scaler = evi_scaler
                st.session_state.evi_metrics = evi_metrics
                st.session_state.le_soil = le_soil
                st.session_state.le_stratum = le_stratum
                
                st.success(f"✅ Crop Health Model trained! R²: {evi_metrics['r2']:.3f}")
                
                # Generate forecasts
                with st.spinner("Generating EVI forecasts..."):
                    evi_forecasts = forecast_evi_trend(
                        df, evi_model, evi_scaler, 
                        evi_metrics['features'], le_soil, le_stratum,
                        forecast_months
                    )
                    st.session_state.evi_forecasts = evi_forecasts
                    if not evi_forecasts.empty:
                        st.success(f"✅ Generated {forecast_months}-month forecasts for {evi_forecasts['stratum_id'].nunique()} strata")
            else:
                st.error("❌ Failed to train Crop Health model")
    
    # Display model info
    st.subheader("Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Features (Temporal features EXCLUDED):**")
        st.write("1. SMI (Soil Moisture Index)")
        st.write("2. NDMI (Normalized Difference Moisture Index)")
        st.write("3. NDRE (Normalized Difference Red Edge)")
        st.write("4. Elevation")
        st.write("5. Soil Texture (encoded)")
        st.write("6. Stratum ID (encoded)")
        
        # st.warning("**EXCLUDED from model:** year, month, month_name, season, date")
    
    with col2:
        # Display model info
        if st.session_state.evi_model is not None:
            metrics = st.session_state.evi_metrics
            
            st.metric("R² Score", f"{metrics['r2']:.4f}")
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
            st.metric("MAE", f"{metrics['mae']:.4f}")
            st.metric("Training Samples", metrics['n_train'])
            st.metric("Test Samples", metrics['n_test'])
            st.metric("Cross-Validation R²", f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            
            # Forecast info
            if st.session_state.evi_forecasts is not None:
                forecast_df = st.session_state.evi_forecasts
                st.metric("Forecast Months", forecast_months)
                st.metric("Forecast Strata", forecast_df['stratum_id'].nunique())
        else:
            st.info("Click 'Train Crop Health Model' button to train the model")
    
    # Model results
    if st.session_state.evi_model is not None:
        st.subheader("Model Analysis")
        
        # Feature importance
        metrics = st.session_state.evi_metrics
        if 'feature_importance' in metrics and metrics['feature_importance']:
            importance_df = pd.DataFrame(
                list(metrics['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance for EVI Prediction',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # EVI Trend Forecasting
        if st.session_state.evi_forecasts is not None and not st.session_state.evi_forecasts.empty:
            st.subheader(f"EVI Trend Forecast ({forecast_months} months)")
            
            # Select stratum for forecast visualization
            forecast_stratum = st.selectbox(
                "Select Stratum for Forecast Visualization",
                options=st.session_state.evi_forecasts['stratum_id'].unique(),
                key='evi_forecast_stratum'
            )
            
            if forecast_stratum:
                # Get historical data
                stratum_data = df[df['stratum_id'] == forecast_stratum].sort_values('date')
                stratum_forecasts = st.session_state.evi_forecasts[
                    st.session_state.evi_forecasts['stratum_id'] == forecast_stratum
                ].sort_values('forecast_date')
                
                # Create combined plot
                fig = go.Figure()
                
                # Historical EVI
                fig.add_trace(go.Scatter(
                    x=stratum_data['date'],
                    y=stratum_data['EVI'],
                    mode='lines+markers',
                    name='Historical EVI',
                    line=dict(color='green', width=2),
                    marker=dict(size=6)
                ))
                
                # Forecast EVI
                fig.add_trace(go.Scatter(
                    x=stratum_forecasts['forecast_date'],
                    y=stratum_forecasts['evi_forecast_smoothed'],
                    mode='lines+markers',
                    name='Forecast EVI',
                    line=dict(color='lightgreen', width=2, dash='dash'),
                    marker=dict(size=6, symbol='diamond')
                ))
                
                # Add thresholds
                fig.add_hline(
                    y=evi_critical,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Critical"
                )
                
                fig.add_hline(
                    y=evi_payout,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Payout"
                )
                
                fig.update_layout(
                    title=f'EVI Trend Forecast for {forecast_stratum}',
                    xaxis_title='Date',
                    yaxis_title='EVI',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Export forecasts
                csv_data = stratum_forecasts.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name=f"evi_forecast_{forecast_stratum}.csv",
                    mime="text/csv"
                )

# Tab 3: SPI Trends Analysis
with tab3:
    st.markdown('<h2 class="section-header">Drought Trend Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate SPI trends
    if st.button("📈 Analyze Drought Trends", type="primary", key="analyze_spi_trends"):
        with st.spinner("Analyzing Drought trends..."):
            spi_trends = analyze_spi_trends(df)
            st.session_state.spi_trends = spi_trends
            st.success(f"✅ Analyzed trends for {len(spi_trends)} strata")
    
    # Display SPI trends
    if st.session_state.spi_trends is not None:
        spi_trends = st.session_state.spi_trends
        
        st.subheader("Drought Trend Analysis by Stratum")
        
        # Display trends table
        st.dataframe(
            spi_trends,
            use_container_width=True,
            height=400
        )
        
        # Time series analysis for selected stratum
        st.subheader("Detailed Trend Analysis")
        
        selected_stratum = st.selectbox(
            "Select Stratum for Detailed Analysis",
            options=spi_trends['stratum_id'].unique(),
            key='spi_stratum_select'
        )
        
        if selected_stratum:
            # Get time series data
            stratum_data = df[df['stratum_id'] == selected_stratum].sort_values('date')
            trend_info = spi_trends[spi_trends['stratum_id'] == selected_stratum].iloc[0]
            
            # Create time series plot with trend line
            fig = go.Figure()
            
            # Add actual SPI values
            fig.add_trace(go.Scatter(
                x=stratum_data['date'],
                y=stratum_data['SPI'],
                mode='lines+markers',
                name='SPI',
                line=dict(color='blue', width=2),
                marker=dict(size=6, opacity=0.7)
            ))
            
            # Add trend line
            if len(stratum_data) >= 2:
                x_numeric = np.arange(len(stratum_data))
                slope = float(trend_info['trend_slope'])
                intercept = stratum_data['SPI'].iloc[0] - slope * x_numeric[0]
                trend_line = slope * x_numeric + intercept
                
                fig.add_trace(go.Scatter(
                    x=stratum_data['date'],
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash', width=2)
                ))
            
            # Add thresholds
            fig.add_hline(
                y=spi_critical,
                line_dash="dash",
                line_color="orange",
                annotation_text="Critical"
            )
            
            fig.add_hline(
                y=spi_payout,
                line_dash="dot",
                line_color="red",
                annotation_text="Payout"
            )
            
            fig.update_layout(
                title=f'SPI Time Series for {selected_stratum}',
                xaxis_title='Date',
                yaxis_title='SPI (Standardized Precipitation Index)',
                hovermode='x unified',
                height=500,
                showlegend=True,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend analysis summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current SPI", trend_info['current_spi'])
            with col2:
                st.metric("Trend Direction", trend_info['trend_category'])
            with col3:
                st.metric("Risk Level", trend_info['risk_level'])
            with col4:
                st.metric("Next Month Forecast", trend_info['next_1_month'])

# Tab 4: SPI Trend Forecasting (Integrated)
with tab4:
    st.markdown('<h2 class="section-header">SPI Trend Forecasting Model (Autoregressive)</h2>', unsafe_allow_html=True)
    
    # Model training
    if train_spi:
        with st.spinner("Training SPI trend forecasting model..."):
            spi_model, spi_metrics, df_processed = train_spi_trend_model(df, model_type)
            
            if spi_model is not None:
                st.session_state.spi_trend_model = spi_model
                st.session_state.spi_metrics = spi_metrics
                st.session_state.spi_processed_data = df_processed
                
                st.success(f"✅ SPI Trend Forecasting Model trained! R²: {spi_metrics['r2']:.3f}, RMSE: {spi_metrics['rmse']:.3f}")
            else:
                st.error("❌ Failed to train SPI trend model")
    
    # Display model info
    st.subheader("Drought Autoregressive Trend Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Features (Pure Time-Series):**")
        st.write("1. SPI Lags 1-12 months (autoregressive)")
        st.write("2. Rolling means (3, 6, 12 months)")
        st.write("3. Rolling standard deviations")
        st.write("4. Seasonal patterns (sine/cosine)")
        st.write("")
        # st.info("📈 **Pure Time-Series Model:** Predicts future SPI based ONLY on historical SPI patterns, trends, and seasonality. No external factors used.")
        # st.warning("**EXCLUDED from model:** rainfall, SMI, elevation, soil texture, stratum")
    
    with col2:
        # Display model info
        if st.session_state.spi_trend_model is not None:
            metrics = st.session_state.spi_metrics
            
            st.metric("R² Score", f"{metrics['r2']:.4f}")
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
            st.metric("Training Samples", metrics['n_samples'])
            st.metric("Time-Series CV RMSE", f"{metrics['rmse']:.4f}")
            
            # Generate forecasts on demand
            if st.button("🔮 Generate Trend Forecasts", type="primary"):
                with st.spinner("Generating recursive forecasts..."):
                    forecasts_list = []
                    for stratum in df['stratum_id'].unique()[:5]:  # Limit for performance
                        stratum_data = st.session_state.spi_processed_data[
                            st.session_state.spi_processed_data['stratum_id'] == stratum
                        ].sort_values('date')
                        
                        if len(stratum_data) > 0:
                            last_row = stratum_data.iloc[-1]
                            future_df = recursive_forecast(
                                st.session_state.spi_trend_model, 
                                last_row, 
                                st.session_state.spi_metrics['features'], 
                                forecast_months
                            )
                            if future_df is not None and not future_df.empty:
                                forecasts_list.append(future_df)
                    
                    if forecasts_list:
                        st.session_state.spi_forecasts = pd.concat(forecasts_list, ignore_index=True)
                        st.success(f"✅ Generated {forecast_months}-month forecasts for {len(forecasts_list)} strata")
                    else:
                        st.error("❌ Failed to generate forecasts")
        else:
            st.info("Click 'Train SPI Model' button in sidebar to train the trend forecasting model")
    
    # Model results
    if st.session_state.spi_trend_model is not None:
        st.subheader("Model Analysis")
        
        # Feature importance
        metrics = st.session_state.spi_metrics
        if 'feature_importance' in metrics and metrics['feature_importance']:
            importance_df = pd.DataFrame(
                list(metrics['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                title="What drives the SPI trend?",
                color='Importance',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Note: 'SPI_lag_1' usually dominates, indicating strong autocorrelation (persistence) in drought conditions.")
        
        # SPI Trend Forecasting
        if st.session_state.spi_forecasts is not None and not st.session_state.spi_forecasts.empty:
            st.subheader(f"Drought Trend Forecast ({forecast_months} months)")
            
            # Select stratum for forecast visualization
            forecast_stratum = st.selectbox(
                "Select Stratum for SPI Trend Forecast Visualization",
                options=st.session_state.spi_forecasts['stratum_id'].unique(),
                key='spi_trend_forecast_stratum'
            )
            
            if forecast_stratum:
                # Get historical data
                stratum_data = df[df['stratum_id'] == forecast_stratum].sort_values('date')
                stratum_forecasts = st.session_state.spi_forecasts[
                    st.session_state.spi_forecasts['stratum_id'] == forecast_stratum
                ].sort_values('forecast_date')
                
                # Risk assessment
                min_forecast_spi = stratum_forecasts['spi_forecast_smoothed'].min()
                risk_status = "VIABLE FOR COMPENSATION" if min_forecast_spi < spi_payout else "NO IMMEDIATE RISK"
                status_color = "red" if min_forecast_spi < spi_payout else "green"
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Historical Data Points", len(stratum_data))
                col2.metric("Model RMSE (Accuracy)", f"{metrics['rmse']:.3f}")
                col3.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold'>{risk_status}</span>", unsafe_allow_html=True)
                
                # Create combined plot
                fig = go.Figure()
                
                # Historical SPI
                fig.add_trace(go.Scatter(
                    x=stratum_data['date'],
                    y=stratum_data['SPI'],
                    mode='lines',
                    name='Historical SPI',
                    line=dict(color='gray', width=1.5)
                ))
                
                # Forecast SPI
                fig.add_trace(go.Scatter(
                    x=stratum_forecasts['forecast_date'],
                    y=stratum_forecasts['spi_forecast_smoothed'],
                    mode='lines+markers',
                    name='Projected Trend (2026)',
                    line=dict(color='#1D976C', width=3)
                ))
                
                # Add SPI categories
                fig.add_hrect(y0=-2.0, y1=-1.5, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Extremely Dry")
                fig.add_hrect(y0=-1.5, y1=-1.0, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Severely Dry")
                fig.add_hrect(y0=-1.0, y1=-0.5, line_width=0, fillcolor="yellow", opacity=0.1, annotation_text="Moderately Dry")
                fig.add_hrect(y0=-0.5, y1=0.5, line_width=0, fillcolor="lightgreen", opacity=0.1, annotation_text="Near Normal")
                fig.add_hrect(y0=0.5, y1=1.0, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Moderately Wet")
                fig.add_hrect(y0=1.0, y1=1.5, line_width=0, fillcolor="blue", opacity=0.1, annotation_text="Very Wet")
                fig.add_hrect(y0=1.5, y1=2.0, line_width=0, fillcolor="darkblue", opacity=0.1, annotation_text="Extremely Wet")
                
                # Add thresholds
                fig.add_hline(
                    y=spi_critical,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Critical"
                )
                
                fig.add_hline(
                    y=spi_payout,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Payout Trigger"
                )
                
                fig.update_layout(
                    title=f'SPI Trend Projection: {forecast_stratum} (Autoregressive ML)',
                    xaxis_title='Timeline',
                    yaxis_title='SPI (Standardized Precipitation Index)',
                    template='plotly_white',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast statistics
                st.subheader("Trend Forecast Analysis")
                
                # Calculate forecast statistics
                avg_forecast = stratum_forecasts['spi_forecast_smoothed'].mean()
                min_forecast = stratum_forecasts['spi_forecast_smoothed'].min()
                max_forecast = stratum_forecasts['spi_forecast_smoothed'].max()
                below_critical = (stratum_forecasts['spi_forecast_smoothed'] < spi_critical).sum()
                below_payout = (stratum_forecasts['spi_forecast_smoothed'] < spi_payout).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Trend Forecast", f"{avg_forecast:.2f}")
                with col2:
                    st.metric("Min Trend Forecast", f"{min_forecast:.2f}")
                with col3:
                    st.metric("Months < Critical", below_critical)
                with col4:
                    st.metric("Months < Payout", below_payout)
                
                # Export forecasts
                csv_data = stratum_forecasts.to_csv(index=False)
                st.download_button(
                    label="📥 Download SPI Trend Forecast CSV",
                    data=csv_data,
                    file_name=f"spi_trend_forecast_{forecast_stratum}.csv",
                    mime="text/csv"
                )

# Tab 5: Risk Map
with tab5:
    st.markdown('<h2 class="section-header">Spatial Risk Assessment Map</h2>', unsafe_allow_html=True)
    
    # Create risk map
    if st.session_state.spi_forecasts is not None:
        st.info(f"Showing risk map based on {forecast_months}-month SPI trend forecasts")
        risk_map = create_risk_map(gdf, df, st.session_state.spi_forecasts)
    else:
        st.info("Showing risk map based on historical SPI data")
        risk_map = create_risk_map(gdf, df)
    
    if risk_map:
        st_folium(risk_map, width=None, height=600)
        
        # Risk legend explanation
        st.subheader("Risk Level Explanation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="risk-indicator risk-high">HIGH RISK</div>', unsafe_allow_html=True)
            st.write("**SPI < -1.5**")
            st.write("Severe drought conditions")
            st.write("High probability of payout")
        
        with col2:
            st.markdown('<div class="risk-indicator risk-medium">MEDIUM RISK</div>', unsafe_allow_html=True)
            st.write("**-1.5 ≤ SPI < -1.0**")
            st.write("Moderate drought")
            st.write("Monitor closely")
        
        with col3:
            st.markdown('<div class="risk-indicator risk-low">LOW RISK</div>', unsafe_allow_html=True)
            st.write("**SPI ≥ -1.0**")
            st.write("Normal or wet conditions")
            st.write("Low probability of payout")

# Tab 6: Insurance Product
with tab6:
    st.markdown('<h2 class="section-header">Dual-Threshold Insurance Product</h2>', unsafe_allow_html=True)
    
    # Create insurance product
    if st.button("💰 Generate Insurance Product", type="primary", use_container_width=True):
        with st.spinner("Creating insurance product..."):
            insurance_product = create_dual_threshold_insurance(
                df, evi_payout, spi_payout, base_premium
            )
            
            st.session_state.insurance_product = insurance_product
            st.success(f"✅ Insurance product created for {len(insurance_product)} strata!")
    
    # Display insurance product
    if st.session_state.insurance_product is not None:
        insurance_df = st.session_state.insurance_product
        
        st.subheader("Insurance Product Parameters")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Category",
                options=['All', 'High', 'Medium', 'Low'],
                index=0,
                key='risk_filter'
            )
        
        with col2:
            min_premium = st.slider(
                "Minimum Premium ($/ha)", 
                0, 300, 0, 10,
                key='min_premium'
            )
        
        with col3:
            max_payout = st.slider(
                "Minimum Max Payout ($/ha)",
                0, 500, 0, 10,
                key='max_payout'
            )
        
        # Apply filters
        filtered_insurance = insurance_df.copy()
        
        if risk_filter != 'All':
            filtered_insurance = filtered_insurance[filtered_insurance['risk_category'] == risk_filter]
        
        # Convert to numeric for filtering
        filtered_insurance['premium_num'] = filtered_insurance['premium_per_ha'].str.replace('$', '').astype(float)
        filtered_insurance['payout_num'] = filtered_insurance['max_payout_per_ha'].str.replace('$', '').astype(float)
        
        filtered_insurance = filtered_insurance[
            (filtered_insurance['premium_num'] >= min_premium) & 
            (filtered_insurance['payout_num'] >= max_payout)
        ]
        
        # Display table
        st.dataframe(
            filtered_insurance.drop(['premium_num', 'payout_num'], axis=1),
            use_container_width=True,
            height=400
        )
        
        # Insurance analysis
        st.subheader("Insurance Product Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            insurance_df['risk_score_num'] = insurance_df['risk_score'].astype(float)
            fig = px.box(
                insurance_df,
                y='risk_score_num',
                title='Risk Score Distribution',
                points='all',
                color_discrete_sequence=['#ff6b6b']
            )
            fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="Low Risk")
            fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
            fig.update_layout(yaxis_title="Risk Score (0-100)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Premium vs Payout
            insurance_df['premium_num'] = insurance_df['premium_per_ha'].str.replace('$', '').astype(float)
            insurance_df['payout_num'] = insurance_df['max_payout_per_ha'].str.replace('$', '').astype(float)
            
            fig = px.scatter(
                insurance_df,
                x='premium_num',
                y='payout_num',
                size='risk_score_num',
                color='risk_category',
                title='Premium vs Maximum Payout',
                hover_name='stratum_id',
                size_max=20,
                labels={
                    'premium_num': 'Premium ($/ha)',
                    'payout_num': 'Max Payout ($/ha)',
                    'risk_score_num': 'Risk Score'
                },
                color_discrete_map={
                    'Low': '#00cc96',
                    'Medium': '#ffa15a',
                    'High': '#ef553b'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export insurance product
        st.subheader("Export Insurance Product")
        
        csv_data = insurance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Insurance Product CSV",
            data=csv_data,
            file_name="insurance_product.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>🌍 Crop Health & Drought Risk Insurance Dashboard</h3>
    <p><b>Study Area:</b> Trans Nzoia County, Kenya</p>
    <p><b>Crop Health Model:</b> EVI = f(SMI, NDMI, NDRE, Elevation, Soil Texture) - Temporal features EXCLUDED</p>
    <p><b>Drought Trend Model:</b> SPI_t+1 = f(SPI_t, SPI_t-1, SPI_t-2, ..., Seasonal Patterns) - Pure autoregressive</p>
    <p><b>Forecasting:</b> {}-month EVI & SPI trend predictions</p>
    <p><b>Map Features:</b> Interactive strata boundaries with automatic zoom and layer controls</p>
    <p><b>Product:</b> Dual-threshold parametric insurance with EVI & SPI triggers</p>
</div>
""".format(forecast_months), unsafe_allow_html=True)
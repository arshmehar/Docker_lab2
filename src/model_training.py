import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ['Pclass', 'Sex', 'Age', 'Fare']

if __name__ == '__main__':
    # Load the Titanic dataset, keep 4 features, drop rows with missing values
    df = pd.read_csv('data/titanic.csv')[FEATURES + ['Survived']].dropna()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    X, y = df[FEATURES].values, df['Survived'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Build a simple TensorFlow model for binary classification
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Save the model and the scaler so serving uses the same preprocessing
    model.save('my_model.keras')
    joblib.dump(sc, 'scaler.pkl')
    print("Model and scaler were trained and saved")

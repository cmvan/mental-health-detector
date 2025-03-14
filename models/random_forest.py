from sklearn.ensemble import RandomForestClassifier
import utils


if __name__ == "__main__":
    # Load data
    train_df, test_df = utils.load_data()
    X_train, X_test = utils.preprocess_text(train_df, test_df)
    y_train, y_test = utils.encode_labels(train_df, test_df)

    # Training
    rf_classifier = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, max_depth=20)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    # Evaluation
    utils.evaluate_model(y_test, y_pred, "Random Forest")

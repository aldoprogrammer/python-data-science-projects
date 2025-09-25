from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data():
    """Load and prepare the gender classification dataset"""
    # [height(cm), weight(kg), shoe_size(EU)]
    X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
         [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
         [181, 85, 43]]

    Y = ['male', 'female', 'female', 'female', 'female',
         'male', 'female', 'female', 'female', 'male',
         'male']
    
    # Convert to DataFrame for better visualization
    feature_names = ['Height (cm)', 'Weight (kg)', 'Shoe Size (EU)']
    df = pd.DataFrame(X, columns=feature_names)
    df['Gender'] = Y
    
    return df, X, Y, feature_names

def visualize_data(df):
    """Create visualizations to understand the data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gender Classification Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Gender distribution
    gender_counts = df['Gender'].value_counts()
    axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                   colors=['lightblue', 'lightpink'], startangle=90)
    axes[0, 0].set_title('Gender Distribution')
    
    # 2. Height vs Weight scatter plot
    for gender in df['Gender'].unique():
        subset = df[df['Gender'] == gender]
        axes[0, 1].scatter(subset['Height (cm)'], subset['Weight (kg)'], 
                          label=gender, alpha=0.7, s=100)
    axes[0, 1].set_xlabel('Height (cm)')
    axes[0, 1].set_ylabel('Weight (kg)')
    axes[0, 1].set_title('Height vs Weight by Gender')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Shoe size distribution
    df.boxplot(column='Shoe Size (EU)', by='Gender', ax=axes[1, 0])
    axes[1, 0].set_title('Shoe Size Distribution by Gender')
    axes[1, 0].set_xlabel('Gender')
    axes[1, 0].set_ylabel('Shoe Size (EU)')
    
    # 4. Feature correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

def train_model(X, Y):
    """Train the Decision Tree classifier"""
    print("Training Decision Tree Classifier...")
    
    # Create and train the model
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X, Y)
    
    # Calculate accuracy on training data
    train_predictions = clf.predict(X)
    train_accuracy = accuracy_score(Y, train_predictions)
    
    print(f"Training Accuracy: {train_accuracy:.2%}")
    
    return clf

def visualize_decision_tree(clf, feature_names):
    """Visualize the decision tree"""
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=feature_names, class_names=['female', 'male'], 
                   filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree for Gender Classification')
    plt.show()

def make_predictions(clf, feature_names):
    """Interactive prediction function"""
    print("\n" + "="*50)
    print("GENDER PREDICTION SYSTEM")
    print("="*50)
    
    # Original prediction
    original_prediction = clf.predict([[190, 70, 43]])
    print(f"Original prediction for [190cm, 70kg, size 43]: {original_prediction[0]}")
    
    # Interactive predictions
    while True:
        print("\nEnter physical characteristics for prediction:")
        print("(Press Enter to skip interactive mode)")
        
        try:
            height = input("Height in cm (e.g., 175): ").strip()
            if not height:
                break
                
            weight = input("Weight in kg (e.g., 70): ").strip()
            shoe_size = input("Shoe size EU (e.g., 42): ").strip()
            
            # Convert to numbers
            height = float(height)
            weight = float(weight)
            shoe_size = float(shoe_size)
            
            # Make prediction
            prediction = clf.predict([[height, weight, shoe_size]])
            
            # Get prediction probability
            probabilities = clf.predict_proba([[height, weight, shoe_size]])
            prob_female = probabilities[0][0]
            prob_male = probabilities[0][1]
            
            print(f"\nPrediction Results:")
            print(f"Input: Height={height}cm, Weight={weight}kg, Shoe Size={shoe_size}")
            print(f"Predicted Gender: {prediction[0].upper()}")
            print(f"Confidence: Female={prob_female:.1%}, Male={prob_male:.1%}")
            
            continue_pred = input("\nMake another prediction? (y/n): ").lower()
            if continue_pred != 'y':
                break
                
        except ValueError:
            print("Please enter valid numeric values!")
        except Exception as e:
            print(f"Error: {e}")

def analyze_feature_importance(clf, feature_names):
    """Analyze which features are most important"""
    importances = clf.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Feature Importance in Gender Classification')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(importances):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("Feature Importance Analysis:")
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.3f} ({importance/sum(importances):.1%})")

def check_environment():
    """Check if required packages are installed"""
    required_packages = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\nTo install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print(f"\nOr install all at once:")
        print(f"pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def main():
    """Main function to run the gender classification project"""
    print("="*60)
    print("        GENDER CLASSIFICATION WITH MACHINE LEARNING")
    print("="*60)
    print("This project predicts gender based on physical characteristics:")
    print("- Height (cm)")
    print("- Weight (kg)") 
    print("- Shoe Size (EU)")
    print("="*60)
    
    # Check environment first
    if not check_environment():
        return
    
    # Load data
    df, X, Y, feature_names = load_data()
    
    print(f"\nDataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {df['Gender'].unique()}")
    
    # Show dataset
    print(f"\nDataset:")
    print(df.to_string(index=False))
    
    # Visualize data
    print(f"\n📊 Generating data visualizations...")
    visualize_data(df)
    
    # Train model
    clf = train_model(X, Y)
    
    # Visualize decision tree
    print(f"\n🌳 Visualizing decision tree...")
    visualize_decision_tree(clf, feature_names)
    
    # Analyze feature importance
    print(f"\n📈 Analyzing feature importance...")
    analyze_feature_importance(clf, feature_names)
    
    # Make predictions
    make_predictions(clf, feature_names)
    
    print(f"\n✅ Project completed successfully!")
    print(f"\nKey Learning Points:")
    print(f"- Decision trees split data based on feature values")
    print(f"- Small datasets can lead to overfitting")
    print(f"- Feature importance helps understand model decisions")
    print(f"- Real-world gender prediction would need much larger datasets")

if __name__ == "__main__":
    main()

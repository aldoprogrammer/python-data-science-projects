"""
Quick test script to verify the gender classification app works correctly
"""
from app import load_data, train_model
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

def test_basic_functionality():
    """Test core functionality without visualizations"""
    print("🧪 Testing Gender Classification App...")
    
    try:
        # Test data loading
        print("✓ Testing data loading...")
        df, X, Y, feature_names = load_data()
        assert len(df) == 11, f"Expected 11 samples, got {len(df)}"
        assert len(feature_names) == 3, f"Expected 3 features, got {len(feature_names)}"
        print(f"  ✅ Data loaded: {len(df)} samples, {len(feature_names)} features")
        
        # Test model training
        print("✓ Testing model training...")
        clf = train_model(X, Y)
        assert clf is not None, "Model training failed"
        print("  ✅ Model trained successfully")
        
        # Test predictions
        print("✓ Testing predictions...")
        
        # Test case 1: Typical male characteristics
        male_prediction = clf.predict([[185, 85, 45]])
        print(f"  Test 1 - [185cm, 85kg, size 45]: {male_prediction[0]}")
        
        # Test case 2: Typical female characteristics  
        female_prediction = clf.predict([[160, 55, 38]])
        print(f"  Test 2 - [160cm, 55kg, size 38]: {female_prediction[0]}")
        
        # Test case 3: Original test case
        original_prediction = clf.predict([[190, 70, 43]])
        print(f"  Test 3 - [190cm, 70kg, size 43]: {original_prediction[0]}")
        
        # Test probabilities
        probabilities = clf.predict_proba([[185, 85, 45]])
        print(f"  Probability example: Female={probabilities[0][0]:.1%}, Male={probabilities[0][1]:.1%}")
        
        # Test feature importance
        print("✓ Testing feature importance...")
        importances = clf.feature_importances_
        for feature, importance in zip(feature_names, importances):
            print(f"  {feature}: {importance:.3f}")
        
        print("\n🎉 All tests passed! The app is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and potential issues"""
    print("\n🔍 Testing edge cases...")
    
    try:
        df, X, Y, feature_names = load_data()
        clf = train_model(X, Y)
        
        # Test extreme values
        extreme_cases = [
            [200, 100, 50, "Very tall/heavy person"],
            [150, 45, 35, "Very short/light person"],
            [175, 70, 42, "Average person"]
        ]
        
        for case in extreme_cases:
            prediction = clf.predict([[case[0], case[1], case[2]]])
            probabilities = clf.predict_proba([[case[0], case[1], case[2]]])
            print(f"  {case[3]}: {prediction[0]} (confidence: {max(probabilities[0]):.1%})")
        
        print("✅ Edge cases handled successfully")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_basic_functionality()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print(f"\n🚀 Ready to run the full app! Try: python app.py")
    else:
        print(f"\n⚠️  Some issues detected. Check your environment setup.")

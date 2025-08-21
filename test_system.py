"""
Quick test to verify the system is working
"""

import requests
import os

def test_system():
    base_url = "http://127.0.0.1:8000"
    
    print("Testing AI Content Detector System...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server not responding")
            return
    except:
        print("❌ Cannot connect to server")
        return
    
    # Test 2: Check if model files exist
    model_files = ["face_detector.pth", "class_to_idx.json"]
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
    
    # Test 3: Test image detection with a sample image
    print("\nTesting image detection...")
    try:
        # Use a sample image from the dataset
        sample_image_path = "dataset/ai/0024267a-6e1e-4fe8-821d-dd9bdf2b9e8e.png"
        if os.path.exists(sample_image_path):
            with open(sample_image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/detect/image", files=files)
                
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Image detection working")
                print(f"   Verdict: {result['verdict']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Modality: {result['modality']}")
            else:
                print(f"❌ Image detection failed: {response.status_code}")
        else:
            print("⚠️  Sample image not found, skipping image test")
    except Exception as e:
        print(f"❌ Image detection error: {e}")
    
    print("\n" + "=" * 50)
    print("System test completed!")

if __name__ == "__main__":
    test_system()

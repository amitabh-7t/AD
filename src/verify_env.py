import tensorflow as tf
import sys
import os

def verify_environment():
    print(f"Python Version: {sys.version}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check for GPU access
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ SUCCESS: {len(gpus)} GPU(s) found!")
        for gpu in gpus:
            print(f"  - Name: {gpu.name}")
            print(f"  - Type: {gpu.device_type}")
    else:
        print("\n❌ WARNING: No GPU found. Code will run on CPU, which is much slower.")
        print("Make sure tensorflow-metal is installed.")

    # Simple Matrix Multiplication Test on GPU
    try:
        print("\nRunning simple matrix multiplication test...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"Result:\n{c.numpy()}")
        print("✅ Matrix multiplication successful on GPU.")
    except Exception as e:
        print(f"❌ Matrix multiplication failed: {e}")

if __name__ == "__main__":
    verify_environment()

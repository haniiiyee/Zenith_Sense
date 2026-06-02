print("--- SYSTEM STARTUP: Initializing Zenith_Sense Benchmark ---")
import tensorflow as tf
import numpy as np
import time
import tracemalloc
import os
import sys

# Force the computer to print immediately (flush buffers)
sys.stdout.reconfigure(encoding='utf-8')

def benchmark_tflite(model_path):
    print(f"--- Benchmarking Target: {model_path} ---")
    
    # 1. SETUP INTERPRETER (Safe Mode for Windows)
    try:
        # We try to disable the default delegates to avoid the XNNPACK crash
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    except AttributeError:
        print("Warning: Using standard interpreter fallback...")
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    # 2. GENERATE DUMMY INPUT
    # Create random numbers to simulate camera data
    if input_details[0]['dtype'] == np.uint8:
        dummy_input = np.random.randint(0, 255, input_shape).astype(np.uint8)
    else:
        dummy_input = np.random.random_sample(input_shape).astype(np.float32)

    # 3. MEASURE RAM (Peak Usage)
    print(">>> Starting Memory Trace...")
    tracemalloc.start()
    
    # Warmup run (wakes up the CPU)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    # 4. MEASURE SPEED (Inference Time)
    print(">>> Running 50 Inferences...")
    start_time = time.time()
    runs = 50
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    end_time = time.time()
    
    # Stop measuring
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 5. GENERATE REPORT
    avg_time = ((end_time - start_time) / runs) * 1000
    peak_ram_mb = peak / (1024 * 1024)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    print(f"\n" + "="*30)
    print(f"   ZENITH_SENSE MISSION REPORT")
    print(f"="*30)
    print(f"Model Name      : zenith_int8.tflite")
    print(f"File Size       : {file_size_mb:.2f} MB")
    print(f"Peak RAM Usage  : {peak_ram_mb:.2f} MB")
    print(f"Inference Speed : {avg_time:.2f} ms")
    print(f"="*30)
    
    if peak_ram_mb < 50:
        print("MISSION STATUS: [PASS] Ready for Space-Grade Hardware.")
    else:
        print("MISSION STATUS: [FAIL] Exceeds Memory Limits.")

# --- THE START BUTTON (Crucial Part) ---
if __name__ == "__main__":
    target_file = 'models/zenith_int8.tflite'
    if os.path.exists(target_file):
        try:
            benchmark_tflite(target_file)
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"ERROR: Could not find model file at {target_file}")
        print("Did you run script 03 successfully?")
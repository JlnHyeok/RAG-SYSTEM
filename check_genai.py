
import google.generativeai as genai
import inspect

print(f"GenAI Version: {genai.__version__}")

try:
    print("Checking GenerationConfig signature...")
    sig = inspect.signature(genai.types.GenerationConfig)
    print(f"Signature: {sig}")
    
    if 'response_mime_type' in sig.parameters:
        print("SUCCESS: response_mime_type is supported.")
    else:
        print("FAILURE: response_mime_type is NOT in parameters.")
        
except Exception as e:
    print(f"Error inspecting signature: {e}")

try:
    config = genai.types.GenerationConfig(temperature=0.1, response_mime_type="application/json")
    print("SUCCESS: Instantiated GenerationConfig with response_mime_type.")
except Exception as e:
    print(f"FAILURE: Could not instantiate: {e}")

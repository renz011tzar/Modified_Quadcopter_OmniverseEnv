import onnxruntime as ort
import numpy as np
import torch

def load_onnx_model(onnx_model_path):
    """
    Loads the ONNX model and creates an inference session.
    """
    ort_session = ort.InferenceSession(onnx_model_path)
    return ort_session

def normalize_input(input_tensor):
    """
    Normalize input similar to the model's internal normalization.
    Replace this function with the actual normalization logic used in the ModelWrapper.
    """
    # Placeholder normalization logic
    # You need to replace this with your actual normalization logic
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    normalized_input = (input_tensor - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
    return normalized_input

def perform_onnx_inference(ort_session, obs_np):
    """
    Runs inference using the ONNX model.
    """
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: obs_np}
    mu, log_std, value = ort_session.run(None, ort_inputs)
    sigma = np.exp(log_std)
    action = mu + sigma * np.random.randn(*mu.shape)
    return mu, log_std, value, action

def main():
    # Path to the exported ONNX model
    onnx_model_path = 'runs/Crazyflie/Crazyflie.onnx'  # Adjust path as necessary

    # Load the ONNX model
    ort_session = load_onnx_model(onnx_model_path)

    # Define your input tensor (the one you provided)
    input_tensor = torch.tensor([-0.2590, -0.1912, -0.0104, 1.0000, -0.6439, 1.0000, -0.7125, 0.3026,
                                 -0.9509, -0.0644, 0.9518, 0.2980, 0.0730, -0.0502, -0.0834, 0.9953,
                                 -0.0361, 0.1627, -0.2713, 0.2853, 0.2477, -4.6207, 3.0000, 3.0000,
                                 0.0000], dtype=torch.float32)

    # Normalize the input (adjust the normalization as needed)
    normalized_input_tensor = normalize_input(input_tensor)

    # Prepare the input for ONNX model (convert to numpy array)
    obs_np = input_tensor.unsqueeze(0).cpu().numpy()  # Shape: (1, 25)

    # Perform inference using the ONNX model
    mu_onnx, log_std_onnx, value_onnx, action_onnx = perform_onnx_inference(ort_session, obs_np)

    # Print the results
    print("ONNX Mu:", mu_onnx)
    print("ONNX Log Std:", log_std_onnx)
    print("ONNX Value:", value_onnx)
    print("ONNX Action:", action_onnx)

if __name__ == "__main__":
    main()

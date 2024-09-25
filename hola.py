import torch
from rl_games.torch_runner import Runner
import rl_games.algos_torch.flatten as flatten
import onnx
import onnxruntime as ort
import numpy as np

# Define your configuration based on your training parameters
config = {
    'params': {
        'seed': 5,
        'algo': {
            'name': 'a2c_continuous'
        },
        'model': {
            'name': 'continuous_a2c_logstd'
        },
        'network': {
            'name': 'actor_critic',
            'separate': False,
            'space': {
                'continuous': {
                    'mu_activation': 'None',
                    'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': 0},
                    'fixed_sigma': True
                }
            },
            'mlp': {
                'units': [256, 256, 128],
                'activation': 'tanh',
                'd2rl': False,
                'initializer': {'name': 'default'},
                'regularizer': {'name': 'None'}
            }
        },
        'load_checkpoint': True,  # Flag to load checkpoint
        'load_path': 'Crazyflie.pth',  # Path to your checkpoint
        'config': {
            'name': 'Crazyflie',
            'full_experiment_name': 'Crazyflie',
            'env_name': 'rlgpu',
            'device': 'cpu',  # or 'cuda'
            'device_name': 'cpu',
            'multi_gpu': False,
            'ppo': True,
            'mixed_precision': False,
            'normalize_input': True,
            'normalize_value': True,
            'num_actors': 1,  # Adjust based on your setup
            'reward_shaper': {'scale_value': 0.01},
            'normalize_advantage': True,
            'gamma': 0.99,
            'tau': 0.95,
            'learning_rate': 1e-4,
            'lr_schedule': 'adaptive',
            'kl_threshold': 0.016,
            'score_to_win': 1000000000,
            'max_epochs': 10000,
            'save_best_after': 50,
            'save_frequency': 50,
            'grad_norm': 1.0,
            'entropy_coef': 0.0,
            'truncate_grads': True,
            'e_clip': 0.2,
            'horizon_length': 16,
            'minibatch_size': 16384,
            'mini_epochs': 8,
            'critic_coef': 2,
            'clip_value': True,
            'seq_length': 4,
            'bounds_loss_coef': 0.0001
        }
    }
}

# Initialize the Runner
runner = Runner()
runner.load(config)

# Create a player (agent) and restore the trained model
agent = runner.create_player()
agent.restore(config['params']['load_path'])

# Define the ModelWrapper class
class ModelWrapper(torch.nn.Module):
    '''
    Wraps the rl_games model to prepare it for ONNX export.
    '''
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_dict):
        # Normalize observations
        input_dict['obs'] = self.model.norm_obs(input_dict['obs'])
        # Get the network outputs
        return self.model.a2c_network(input_dict)

# Prepare dummy inputs for tracing
sensor_input = torch.tensor([
    [
        1.0627e-03,  5.9856e-04,  1.4635e+00,  # Position error
        0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  # Previous output
        9.9995e-01, -1.9831e-04, -1.0184e-02,  # rot_x
        1.2867e-04,  9.9998e-01, -6.8377e-03,  # rot_y
        1.0185e-02,  6.8361e-03,  9.9992e-01,  # rot_z
        -2.9126e-03,  1.1537e-03, -1.0809e-02,  # Linear velocity
        -1.7453e-02, -1.7453e-02,  0.0000e+00,  # Angular velocity
        1.0000e+00,  1.0000e+00,  0.0000e+00   # State variables
    ]
], dtype=torch.float32).to(agent.device)

inputs = {
    'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
    'rnn_states' : agent.states,
}

# Wrap the model
wrapper = ModelWrapper(agent.model)

# Initialize the TracingAdapter
adapter = flatten.TracingAdapter(wrapper, inputs, allow_non_tensor=True)

# Trace the model
with torch.no_grad():
    traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
    flattened_outputs = traced(*adapter.flattened_inputs)
    print(flattened_outputs)

# Define input and output names
input_names = ['obs']
output_names = ['mu', 'log_std', 'value']

# Export the model to ONNX
torch.onnx.export(
    traced, 
    *adapter.flattened_inputs, 
    "my_model.onnx", 
    verbose=True, 
    input_names=input_names, 
    output_names=output_names,
    opset_version=11,  # Ensure compatibility with ONNX Runtime
    dynamic_axes={
        'obs': {0: 'batch_size'},        # Variable batch size
        'mu': {0: 'batch_size'},
        'log_std': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }
)

# Verify the ONNX model
onnx_model = onnx.load("my_model.onnx")
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
except onnx.checker.ValidationError as e:
    print("ONNX model is invalid:", e)

# Create an ONNX Runtime session
ort_session = ort.InferenceSession("my_model.onnx")

# Function to convert torch tensor to numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Prepare inputs for ONNX Runtime
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sensor_input)}

# Run inference with ONNX
ort_outputs = ort_session.run(None, ort_inputs)
mu_onnx, log_std_onnx, value_onnx = ort_outputs

# Convert log_std to sigma
sigma_onnx = np.exp(log_std_onnx)

# Denormalize mu if necessary
mu_onnx = (mu_onnx + 1) / 2

print("ONNX Mu:", mu_onnx)
print("ONNX Sigma:", sigma_onnx)
print("ONNX Value:", value_onnx)

# Run inference with PyTorch model for comparison
with torch.no_grad():
    # Assuming your model's forward method returns mu, sigma, value
    mu_pytorch, sigma_pytorch, value_pytorch = agent.model.a2c_network(sensor_input)
    mu_pytorch = (mu_pytorch + 1) / 2  # Denormalize if applicable

# Convert PyTorch outputs to NumPy
mu_pytorch_np = to_numpy(mu_pytorch)
sigma_pytorch_np = to_numpy(sigma_pytorch)
value_pytorch_np = to_numpy(value_pytorch)

print("\nPyTorch Mu:", mu_pytorch_np)
print("PyTorch Sigma:", sigma_pytorch_np)
print("PyTorch Value:", value_pytorch_np)

# Compare the outputs
print("\nMu Match:", np.allclose(mu_pytorch_np, mu_onnx, atol=1e-5))
print("Sigma Match:", np.allclose(sigma_pytorch_np, sigma_onnx, atol=1e-5))
print("Value Match:", np.allclose(value_pytorch_np, value_onnx, atol=1e-5))

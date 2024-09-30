import os
import torch
import datetime



# Log a message to a specified log file with a timestamp
def log_to_file(log_file, message):
    """
    Logs a message to a specified log file with a timestamp.

    Parameters:
    - log_file: Path to the log file.
    - message: Message to log.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"{timestamp}: {message}"
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)


# Save the optimizer state to a specified directory
def save_optimizer_state(optimizer, checkpoint_dir):
    """
    Saves the optimizer state to a specified directory.

    Parameters:
    - optimizer: The optimizer whose state is to be saved.
    - checkpoint_dir: Directory where the optimizer state will be saved.
    """
    optimizer_state_path = os.path.join(checkpoint_dir, 'optimizer_state.pth')
    torch.save(optimizer.state_dict(), optimizer_state_path)
    
    
    
    
# Load the optimizer state from a specified directory
def load_optimizer_state(checkpoint_dir):
    """
    Loads the optimizer state from a specified directory.

    Parameters:
    - checkpoint_dir: Directory from where the optimizer state is to be loaded.

    Returns:
    - The loaded optimizer state if available, None otherwise.
    """
    optimizer_state_path = os.path.join(checkpoint_dir, 'optimizer_state.pth')
    if os.path.exists(optimizer_state_path):
        return torch.load(optimizer_state_path)
    else:
        return None
    
# Save the model if it has a better F1 score than the best one so far    
def save_model_if_better(current_f1_score, best_f1_score, model, model_save_path):
  """Save the model if it has a better F1 score than the best one so far."""
  if current_f1_score > best_f1_score:
    torch.save(model.state_dict(), model_save_path)
    print(f"New best model saved with F1 score: {current_f1_score}")
    return current_f1_score
  return best_f1_score

# Load the latest model checkpoint if exists, otherwise return empty model
def load_latest_model(model, model_save_path):
  """Load the latest model checkpoint if exists, otherwise return empty model."""
  if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f"Loaded model from: {model_save_path}")
  return model

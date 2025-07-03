# utils/visualizations.py
import matplotlib.pyplot as plt

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    """
    Plots two metrics from the model training history.

    Args:
        model_training_history (dict): A dictionary containing training history
                                       (e.g., {'train_loss': [...], 'val_loss': [...]}).
        metric_name_1 (str): Name of the first metric to plot (e.g., 'train_loss').
        metric_name_2 (str): Name of the second metric to plot (e.g., 'val_loss').
        plot_name (str): Title for the plot.
    """
    # model_training_history.history là định dạng Keras.
    # Trong PyTorch, bạn có thể truyền trực tiếp dict history từ Trainer.
    metric_value_1 = model_training_history[metric_name_1]
    metric_value_2 = model_training_history[metric_name_2]
    
    # Get the Epochs Count
    epochs = range(len(metric_value_1))

    plt.figure(figsize=(10, 6)) # Tùy chỉnh kích thước biểu đồ
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'orange', label=metric_name_2)

    plt.title(str(plot_name))
    plt.xlabel("Epochs") # Thêm nhãn trục x
    plt.ylabel("Metric Value") # Thêm nhãn trục y
    plt.legend()
    plt.grid(True) # Thêm lưới
    plt.show() # Hiển thị biểu đồ (hoặc plt.savefig() để lưu)

def plot_combined_metrics(model_training_history):
    """
    Plots common training and validation metrics.

    Args:
        model_training_history (dict): A dictionary containing training history.
    """
    # Plot Loss
    plot_metric(model_training_history, 'train_loss', 'val_loss', 'Training and Validation Loss')

    # Plot Accuracy
    plot_metric(model_training_history, 'train_accuracy', 'val_accuracy', 'Training and Validation Accuracy')
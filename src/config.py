IMAGE_TRAIN_PATH = r"dataset/images/images/image_train"
IMAGE_TEST_PATH = r"dataset/images/images/image_test"
IMAGE_TRAIN_OUTPUT = "outputs/features/image/train_embeddings.npz"
IMAGE_TEST_OUTPUT  = "outputs/features/image/test_embeddings.npz"
X_TEXT_TRAIN_PATH = "dataset/images/images/X_train_update.csv"
Y_TEXT_TRAIN_PATH = "dataset/images/images/Y_train_CVw08PX.csv"
X_TEXT_TEST_PATH  = "dataset/images/images/X_test_update.csv"
TEXT_TRAIN_OUTPUT = "outputs/features/text/train_text_embeddings.npz"
TEXT_TEST_OUTPUT  = "outputs/features/text/test_text_embeddings.npz"
TEXT_MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
TRAIN_MULTIMODAL_OUTPUT = "outputs/features/multimodal/train_multimodal_embeddings.npz"
MULTIMODAL_CLUSTER_OUTPUT = "outputs/features/multimodal/clustered_multimodal_embeddings.npz"
MODEL_OUTPUT = "outputs/model/pseudo_label_classifier.pth"
CSV_OUTPUT = "outputs/csv/clustering_result.csv"
EPOCHS = 20
LR = 1e-3
RANDOM_STATE = 42
BATCH_SIZE = 16
NUM_WORKERS = 4

if __name__ == "__main__":
    import os
    OUTPUT_PATHS = [
        IMAGE_TRAIN_OUTPUT,
        IMAGE_TEST_OUTPUT,
        TEXT_TRAIN_OUTPUT,
        TEXT_TEST_OUTPUT,
        TRAIN_MULTIMODAL_OUTPUT,
        MULTIMODAL_CLUSTER_OUTPUT,
        MODEL_OUTPUT,
        CSV_OUTPUT,
    ]

    for path in OUTPUT_PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
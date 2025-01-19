# README: LLaMA 3.1/3.2 Fine-tuning with Unsloth

## Project Overview
This project provides an end-to-end solution for fine-tuning LLaMA 3.1 and LLaMA 3.2 models using the **Unsloth** library for accelerated training. The objective is to fine-tune these models on domain-specific tasks using a CSV file containing a **Question and Answer (QA)** format. The project is specifically designed for **Advanced Named Entity Recognition (NER)** with entity extraction capabilities. After fine-tuning, the model can be used to build domain-specific chatbots or integrated into AI agents. remove the pip install lines iif anyone wants to run it directly. Also, setup a environment using pyenv and install the dependancies using requirements.txt

## Features
- **Fine-tuning LLaMA models:** Supports both LLaMA 3.1 and 3.2 models.
- **Efficient Training:** Utilizes Unsloth for faster processing.
- **QA-based Learning:** Accepts any Question-Answer formatted CSV.
- **Domain-Specific Chatbots:** Train models tailored to specific needs.
- **NER Capabilities:** Extracts name and entity data effectively.
- **Colab Compatibility:** Designed to leverage Google Colab's GPU for training.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/llama-finetune.git
   cd llama-finetune
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Upload your QA CSV file to the project directory.

## How to Run
The project can be executed using Google Colab for GPU acceleration.

1. Open the `colab_finetune.ipynb` notebook in Google Colab.
2. Upload your Question-Answer CSV file.
3. Run the cells step-by-step to preprocess data, fine-tune the model, and save it.
4. Download the trained model for further use in chatbot applications.

Alternatively, to run locally:
```bash
python ice_dev_5_merged_training_3.py --input data/qa_data.csv --output models/finetuned_llama
```

If anyone wants to try the project on Kaggle, visit:
[Kaggle Notebook](https://www.kaggle.com/code/icedev/ice-dev-demo)

## Dependencies
The following dependencies are required:
- **Python 3.x**
- **Hugging Face Transformers** – For model training and inference.
- **Unsloth** – Efficient training and optimization.
- **TRL (Transformers Reinforcement Learning)** – Model reinforcement learning.
- **Torch** – Deep learning framework.
- **Pandas** – Data processing and manipulation.
- **Datasets** – Data management and processing.
- **Weights and Biases (wandb)** – Experiment tracking and visualization.

Install missing dependencies using:
```bash
pip install transformers unsloth trl torch pandas datasets wandb
```

## File Structure
```
|-- ice_dev_5_merged_training_3.py   # Main fine-tuning script
|-- colab_finetune.ipynb             # Google Colab notebook
|-- data/                            # QA dataset storage
|-- models/                          # Trained model output
|-- requirements.txt                 # Dependency list
|-- README.md                        # Project documentation
```

## Usage
- Modify the script to change model parameters and dataset paths.
- Ensure that the dataset follows the Question-Answer format.
- Leverage Google Colab's free GPU for faster training.

## Author
Developed by Siraj Us Salekin.

## License
This project is licensed under the MIT License.

## Contact
For queries, reach out to: sirajussalekinrupak@example.com


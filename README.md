# Synthetic Data Generation by Supervised Neural Gas Network for Physiological Emotion Recognition Data 

This repository contains the implementation of a Supervised Neural Gas (SNG) network designed for synthetic data generation to address data scarcity in the field of emotion recognition using physiological signals. Our approach utilizes physiological signals such as EEG, ECG, and GSR to generate synthetic datasets that closely mimic real-world distributions.

## Key Features
- **Efficient Synthetic Data Generation**: Leveraging the Supervised Neural Gas network, our method generates synthetic instances that effectively reflect the intrinsic patterns of physiological emotion data.
- **Robust Emotion Recognition**: Supports robust model development and generalization across different populations and environments.
- **Fast Processing**: Demonstrated to provide significant improvements in processing time compared to established models like Conditional VAE, Conditional GAN, and others.

## Installation
To set up the project environment:
git clone https://github.com/SeyedMuhammadHosseinMousavi/Synthetic-Data-Generation-by-Supervised-Neural-Gas-Network
cd Synthetic-Data-Generation-by-Supervised-Neural-Gas-Network
pip install -r requirements.txt

## Usage
Run the main script to start the data generation process:
python generate_data.py

## Datasets
This project uses the following datasets for validation:
- BRAINWAVE EEG DATASET
- Emotional Status Determination using Physiological Parameters Data Set (ESD)

Both datasets are used to demonstrate the effectiveness of the synthetic data in enhancing emotion recognition systems.

## Results
Our method achieved competitive performance in terms of accuracy, precision, and recall against several benchmarks. Detailed results and comparisons can be found in the `results` folder.

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Contact
S. Muhammad Hossein Mousavi - s.muhammad.hossein.mousavi@gmail.com

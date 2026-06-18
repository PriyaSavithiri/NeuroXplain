# NeuroXplain

NeuroXplain is a brain tumor MRI classification project built using deep learning and explainable AI.

The project compares different deep learning models and uses explanation methods like Grad-CAM, Grad-CAM++, and LIME to show which parts of the MRI image influenced the model’s prediction. The project also includes a simple Streamlit interface for testing MRI images and viewing explanations.

## Project Aim

The main aim of this project is to make brain tumor classification more understandable, not just accurate. Instead of only giving a prediction, the project also shows visual explanations to help understand why the model made that decision.

## What This Project Includes

- Brain tumor MRI image classification
- Multiple deep learning models
- Grad-CAM and Grad-CAM++ visualizations
- LIME explanations
- Streamlit-based interface
- Result analysis and project report

## Project Structure

```text
NeuroXplain/
│
├── Code/              # Jupyter notebooks and source code
├── CSV/               # CSV files used for results and analysis
├── Report/            # Final report and documentation
├── test-gradcam/      # Grad-CAM testing files
└── README.md
```

## How to Run
### Clone the repository:

```text
git clone https://github.com/YOUR-USERNAME/NeuroXplain.git
```
### Open the project folder:
```text
cd NeuroXplain
```

### Open the Jupyter Notebook files inside the Code/ folder.
Run the notebook cells one by one.

There is no separate requirements file included. Install any missing Python libraries manually when the notebook asks for them.

## Technologies Used

```text
- Python
- Jupyter Notebook
- Deep Learning
- PyTorch
- Grad-CAM
- LIME
- Streamlit
- MRI Image Analysis
```

## Contributors
- Priya Savithiri Baskaran
- Felix Kurian
- Karthika Deepa Kasirajan Rajam
  
## Note
This project was created for academic learning and research purposes. It is not intended for real clinical diagnosis or medical decision-making.

## License
This project is open-source for learning and educational use.

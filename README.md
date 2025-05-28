# Medical Text Processing System

A comprehensive system for processing and analyzing medical text data, featuring dual-task capabilities for medical text tagging and summarization.

## Features

- Medical text tagging and classification
- Medical text summarization
- Web interface for easy interaction
- Support for PDF and DOCX file processing
- Comprehensive evaluation metrics
- Optimized model architecture

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-text-processing.git
cd medical-text-processing
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Use the web interface to:
   - Enter medical text directly
   - Upload PDF or DOCX files
   - View processing results

## Project Structure

```
medical-text-processing/
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── PROJECT_DOCUMENTATION.md  # Detailed documentation
├── templates/            # HTML templates
│   └── index.html       # Main web interface
└── .gitignore           # Git ignore file
```

## Documentation

For detailed documentation about the project, including:
- System architecture
- Data processing pipeline
- Model architecture
- Training configuration
- Performance metrics
- Best practices

Please refer to [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 
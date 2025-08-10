# 🤝 Contributing to AI Content Detector

Thank you for your interest in contributing to the AI Content Detector project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- Basic knowledge of AI/ML concepts

### Development Environment Setup
1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TrueSightQ-Multimodal-AI-Generated-Content-Detection-with-Quantum-Techniques-and-Web3-Integration.git
   cd TrueSightQ-Multimodal-AI-Generated-Content-Detection-with-Quantum-Techniques-and-Web3-Integration
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Yogeshyogi007/TrueSightQ-Multimodal-AI-Generated-Content-Detection-with-Quantum-Techniques-and-Web3-Integration.git
   ```
4. **Set up the development environment**:
   ```bash
   # Backend
   cd backend
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/your-bug-description
```

### 2. Make Your Changes
- Write clear, well-documented code
- Follow the existing code style and conventions
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd ../frontend
npm test

# Run the application to test manually
cd ../backend
uvicorn app.main:app --reload

cd ../frontend
npm run dev
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add new AI detection algorithm

- Implemented advanced DCT analysis
- Added support for new image formats
- Updated documentation
- Added unit tests"
```

**Commit Message Format:**
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Screenshots if UI changes
- Test results
- Any breaking changes

## 📝 Code Style Guidelines

### Python (Backend)
- Follow **PEP 8** style guide
- Use **type hints** for function parameters and return values
- Maximum line length: **88 characters** (Black formatter)
- Use **docstrings** for all functions and classes

```python
def detect_ai_content(
    image_path: str,
    confidence_threshold: float = 0.7
) -> Tuple[str, float]:
    """
    Detect AI-generated content in an image.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        Tuple of (verdict, confidence_score)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If confidence threshold is invalid
    """
    # Implementation here
    pass
```

### JavaScript/TypeScript (Frontend)
- Use **ESLint** and **Prettier** for formatting
- Follow **Next.js** best practices
- Use **TypeScript** for type safety
- Component names in **PascalCase**

```typescript
interface DetectionResult {
  verdict: 'ai' | 'real';
  confidence: number;
  processingTime: number;
}

const DetectionComponent: React.FC<{ result: DetectionResult }> = ({ result }) => {
  // Component implementation
};
```

### General Guidelines
- **Write self-documenting code** with clear variable names
- **Add comments** for complex logic
- **Keep functions small** and focused
- **Use meaningful commit messages**

## 🧪 Testing Guidelines

### Backend Testing
- **Unit tests** for all functions and classes
- **Integration tests** for API endpoints
- **Test coverage** should be >80%
- Use **pytest** framework

```python
import pytest
from app.models.image_detector import ImageDetector

def test_image_detector_initialization():
    """Test ImageDetector class initialization."""
    detector = ImageDetector()
    assert detector is not None
    assert hasattr(detector, 'detect')

def test_detection_with_valid_image():
    """Test detection with a valid image file."""
    detector = ImageDetector()
    result = detector.detect("tests/fixtures/real_image.jpg")
    assert result['verdict'] in ['ai', 'real']
    assert 0 <= result['confidence'] <= 1
```

### Frontend Testing
- **Component tests** using React Testing Library
- **Integration tests** for user workflows
- **E2E tests** for critical paths
- Use **Jest** and **Cypress**

```typescript
import { render, screen } from '@testing-library/react';
import DetectionComponent from '../DetectionComponent';

describe('DetectionComponent', () => {
  it('renders detection result correctly', () => {
    const mockResult = {
      verdict: 'ai' as const,
      confidence: 0.85,
      processingTime: 150
    };
    
    render(<DetectionComponent result={mockResult} />);
    
    expect(screen.getByText('AI-Generated')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
  });
});
```

## 📚 Documentation

### Code Documentation
- **Docstrings** for all public functions and classes
- **Inline comments** for complex logic
- **Type hints** for better code understanding

### API Documentation
- **OpenAPI/Swagger** specifications
- **Example requests/responses**
- **Error codes and messages**

### User Documentation
- **README updates** for new features
- **Usage examples** and tutorials
- **Troubleshooting guides**

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Environment details** (OS, Python version, etc.)
5. **Screenshots or error logs**
6. **Minimal reproduction code**

## 💡 Feature Requests

For feature requests:

1. **Describe the feature** clearly
2. **Explain the use case** and benefits
3. **Provide examples** if possible
4. **Consider implementation** complexity
5. **Check if similar features** already exist

## 🔒 Security

- **Never commit** API keys or secrets
- **Report security issues** privately to maintainers
- **Follow security best practices** in code
- **Validate all inputs** and sanitize outputs

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] **Code follows** style guidelines
- [ ] **Tests pass** and coverage is maintained
- [ ] **Documentation is updated**
- [ ] **No breaking changes** (or clearly documented)
- [ ] **Commit messages** are clear and follow convention
- [ ] **PR description** explains changes clearly

## 🎯 Areas for Contribution

### High Priority
- **Model improvements** and accuracy enhancements
- **Performance optimizations**
- **Bug fixes** and stability improvements
- **Documentation** improvements

### Medium Priority
- **New detection algorithms**
- **UI/UX improvements**
- **Additional file format support**
- **Testing coverage** improvements

### Low Priority
- **Code refactoring**
- **Minor UI tweaks**
- **Documentation updates**
- **Performance monitoring**

## 🏆 Recognition

Contributors will be recognized through:

- **GitHub contributors** list
- **Release notes** mentions
- **Contributor spotlight** in documentation
- **Special thanks** in project acknowledgments

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Wiki**: For detailed documentation
- **Email**: For security issues or private matters

## 🙏 Thank You

Thank you for contributing to the AI Content Detector project! Your contributions help make AI content detection more accessible and accurate for everyone.

---

**Happy coding! 🚀**

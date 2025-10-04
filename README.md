# MediSense - Medical Trends Prediction System

A comprehensive full-stack medical trends prediction system for campus clinic management, featuring Django REST API backend and React frontend with advanced analytics and machine learning capabilities.

## Architecture

- **Backend**: Django REST API with JWT authentication
- **Frontend**: React.js with Material-UI and Nivo charts
- **Database**: MySQL with comprehensive medical data models
- **ML Pipeline**: XGBoost, Random Forest for visit prediction and symptom classification
- **Analytics**: Streamlit integration for advanced data visualization

## Features

### Authentication & User Management
- JWT-based authentication system
- Role-based access control (Admin, Staff)
- User profile management with avatar upload
- Activity logging and audit trails

### Patient & Symptom Management
- Comprehensive patient registration system
- Advanced symptom logging with categorization
- Real-time symptom trend analysis
- Department-wise visit tracking
- Export functionality (Excel, PDF)

### Predictive Analytics
- Visit prediction models (Binary classification, Regression)
- Dominant symptom classification
- Environmental data integration (weather, air quality)
- Academic period correlation analysis
- Interactive data preprocessing tools

### Data Visualization
- Interactive dashboards with real-time charts
- Campus heatmap for department visits
- Time-series analysis of symptom trends
- Comprehensive analytics reports
- Model performance visualization

### System Features
- RESTful API with comprehensive documentation
- Database backup and restore functionality
- Responsive React frontend
- CORS-enabled for frontend-backend communication
- Comprehensive error handling and validation

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MySQL 8.0+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/brynyz/medisense-prototype.git
cd medisense
```
### 2. Backend Setup

#### Create Virtual Environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```
#### Configure Environment
Create a `.env` file in the backend directory:
```env
SECRET_KEY=your_secret_key_here
DEBUG=True
DATABASE_URL=mysql://username:password@localhost:3306/medisense_db
EMAIL_HOST_PASSWORD=your_email_password
```
#### Database Setup
```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
### 3. Frontend Setup

#### Install Dependencies
```bash
cd ../frontend/react-admin
npm install
```
#### Start Development Server
```bash
npm start
```
## Project Structure

```
medisense/
├── backend/                    # Django REST API
│   ├── accounts/              # User authentication & management
│   ├── patients/              # Patient & symptom management
│   ├── medisense/            # Main Django settings
│   ├── data/                 # ML datasets and processing scripts
│   ├── models/               # Trained ML models
│   ├── streamlit_app/        # Streamlit analytics interface
│   └── requirements.txt      # Python dependencies
├── frontend/
│   └── react-admin/          # React frontend application
│       ├── src/
│       │   ├── components/   # Reusable UI components
│       │   ├── scenes/       # Page components
│       │   ├── services/     # API integration
│       │   └── contexts/     # React contexts
│       └── package.json      # Node.js dependencies
└── README.md
```

## API Endpoints

### Authentication
- `POST /api/auth/login/` - User login
- `POST /api/auth/register/` - User registration
- `GET /api/auth/user/` - Get current user
- `PUT /api/auth/profile/update/` - Update user profile

### Patients & Symptoms
- `GET /api/patients/patients/` - List patients
- `POST /api/patients/patients/` - Create patient
- `GET /api/patients/symptoms/` - List symptom logs
- `POST /api/patients/symptoms/` - Create symptom log
- `GET /api/patients/symptoms/chart-data/` - Chart data for trends
- `GET /api/patients/symptoms/department-visits/` - Department visit counts

### Documentation
- `GET /api/docs/` - Swagger UI documentation
- `GET /api/redoc/` - ReDoc documentation
- `GET /api/schema/` - OpenAPI schema

## Machine Learning Models

### Visit Prediction
- **Binary Classification**: Predicts visit/no-visit days (92.4% accuracy)
- **Regression**: Predicts visit counts with environmental factors
- **Features**: Weather data, academic periods, historical patterns

### Symptom Classification
- **Multi-class Classification**: Categorizes dominant symptoms
- **Categories**: Respiratory, Digestive, Pain, Neurological, etc.
- **Accuracy**: 83.3% with XGBoost classifier

## Development

### Backend Development
```bash
cd backend
python manage.py runserver  # Runs on http://localhost:8000
```
### Frontend Development
```bash
cd frontend/react-admin
npm start  # Runs on http://localhost:3000
```
### API Documentation
- Swagger UI: http://localhost:8000/api/docs/
- ReDoc: http://localhost:8000/api/redoc/

## Testing

### Backend Tests
```bash
cd backend
python manage.py test
```
### Frontend Tests
```bash
cd frontend/react-admin
npm test
```
## Deployment

### Backend (Django)
- Configure production settings
- Set up PostgreSQL/MySQL database
- Use Gunicorn + Nginx for production
- Set environment variables for security

### Frontend (React)
- Build production bundle: `npm run build`
- Deploy to Netlify, Vercel, or static hosting
- Configure API base URL for production

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of a thesis research on medical trends prediction systems.

## Author

**Brylle Nyel Manaog**
- GitHub: [@brynyz](https://github.com/brynyz)
- Project: Medical Trends Prediction System Thesis

---

© 2025 MediSense - Campus Medical Trends Prediction System

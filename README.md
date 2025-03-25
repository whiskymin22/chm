# CHM Project

## Backend

The backend is built with FastAPI and can be started using Docker Compose.

## Frontend

The frontend is built with Streamlit. To start the frontend, use Docker Compose.

## Running the Project

To start the entire project, run:

```sh
docker-compose up --build
```

The backend will be available at `http://localhost:8000` and the frontend at `http://localhost:8501`.





The backend folder is structured to handle the core functionality of the application, including API endpoints, database interactions, and machine learning model inference. Here's a breakdown of how it works and connects to other components:

---

### **Folder Structure Overview**
```
backend/
├── app/
│   ├── api/                # API endpoints
│   │   └── v1/             # Versioned API
│   │       └── endpoints/  # Individual endpoint modules
│   ├── db/                 # Database connection and models
│   ├── models/             # Machine learning models (e.g., CRNN)
│   ├── main.py             # Entry point for the FastAPI app
│   └── core/               # Core utilities (e.g., configuration)
├── requirements.txt        # Backend dependencies
├── Dockerfile              # Docker configuration for backend
└── alembic/                # Database migrations
```

---

### **Key Components and Their Roles**

#### 1. **`app/main.py`**
- **Purpose**: Entry point for the FastAPI application.
- **Responsibilities**:
  - Initializes the FastAPI app.
  - Configures middleware (e.g., CORS).
  - Sets up global exception handling.
  - Includes API routers (e.g., `/api/v1`).
- **Connections**:
  - Imports and includes routers from `app/api/v1/endpoints`.
  - Handles requests from the frontend or external clients.

---

#### 2. **`app/api/v1/endpoints/`**
- **Purpose**: Contains individual API endpoint modules.
- **Responsibilities**:
  - Defines routes for specific functionalities (e.g., OCR, health checks).
  - Handles HTTP requests and responses.
  - Uses dependencies like database sessions or ML models.
- **Connections**:
  - Imports the CRNN model from `app/models/crnn.py`.
  - Uses database sessions from `app/db/session.py`.

Example: `ocr.py`
- Handles image uploads.
- Preprocesses the image and passes it to the CRNN model.
- Returns predictions as a JSON response.

---

#### 3. **`app/models/`**
- **Purpose**: Contains machine learning models (e.g., CRNN).
- **Responsibilities**:
  - Defines the architecture of ML models.
  - Handles model loading, saving, and inference.
- **Connections**:
  - Used by API endpoints (e.g., OCR endpoint).
  - Can be extended to include other ML models.

Example: `crnn.py`
- Implements the CRNN model for OCR.
- Provides methods for inference and model state management.

---

#### 4. **`app/db/`**
- **Purpose**: Manages database connections and models.
- **Responsibilities**:
  - Establishes a connection to the PostgreSQL database.
  - Provides session management for database operations.
  - Defines database models (if any).
- **Connections**:
  - Used by API endpoints to interact with the database.
  - Configured using environment variables (e.g., `DATABASE_URL`).

Example: `session.py`
- Creates an asynchronous database engine.
- Provides a dependency (`get_db`) for database sessions.

---

#### 5. **`alembic/`**
- **Purpose**: Handles database migrations.
- **Responsibilities**:
  - Tracks changes to the database schema.
  - Applies migrations to keep the database schema up-to-date.
- **Connections**:
  - Works with `app/db/session.py` to manage the database schema.

---

#### 6. **`requirements.txt`**
- **Purpose**: Lists Python dependencies for the backend.
- **Responsibilities**:
  - Ensures all required libraries (e.g., FastAPI, SQLAlchemy, PyTorch) are installed.
- **Connections**:
  - Used during Docker image build or local setup.

---

### **How It All Connects**

1. **Frontend → Backend**:
   - The frontend sends HTTP requests (e.g., image uploads) to the backend API endpoints.
   - Example: The OCR endpoint processes an image and returns predictions.

2. **Backend → Database**:
   - API endpoints use database sessions to store or retrieve data.
   - Example: Storing OCR results or user data.

3. **Backend → ML Models**:
   - API endpoints load and use ML models for inference.
   - Example: The CRNN model is used to process images and generate predictions.

4. **Backend → External Services**:
   - The backend can interact with external APIs or services if needed.
   - Example: Sending notifications or fetching external data.

---

### **Request Flow Example**
1. **Frontend**: Sends an image to the `/api/v1/ocr` endpoint.
2. **API Endpoint**: 
   - Validates the image.
   - Preprocesses it and passes it to the CRNN model.
   - Returns predictions as a JSON response.
3. **Database** (Optional): Stores the OCR results for future use.

---

This modular structure ensures:
- **Scalability**: Easy to add new features or endpoints.
- **Maintainability**: Clear separation of concerns.
- **Reusability**: Components like the CRNN model or database sessions can be reused across endpoints.
# Labo5: MLOps - End-to-End ML Pipeline with GitHub Actions, Azure ML & Kubernetes

## **What is This Lab About?** 🎯

This lab demonstrates a **complete MLOps (Machine Learning Operations) pipeline** - a production-ready system that automatically trains, packages, and deploys a machine learning model. The lab uses:

- **GitHub Actions** - Automated CI/CD pipeline
- **Azure Machine Learning** - Cloud-based ML training
- **Docker** - Container packaging
- **Kubernetes** - Deployment orchestration

**Goal**: Build a system where pushing code to GitHub automatically trains a model, packages it, and deploys it to production. ✅

---

## **What Does the Pipeline Do?** 🔄

### **Overview**
```
Code Push → Train Model → Package → Deploy to Production → API Ready
```

**Step by step:**
1. **Push code** to GitHub main branch
2. **GitHub Actions triggers** → Automatically runs workflow
3. **Train on Azure ML** → Process 3 datasets (pandas, cats, dogs images)
4. **Download model** → Get trained model from Azure
5. **Build Docker image** → Package with FastAPI server
6. **Deploy to Kubernetes** → Run as scalable API service
7. **API is live** → Accept predictions from users

---

## **What We Built** 🛠️

### **1. GitHub Actions Workflow** (`.github/workflows/azure-ml-workflow.yml`)
**Purpose**: Automates everything from training to deployment

**3 Jobs:**

#### **Job 1: `azure-pipeline` (Training on Azure)**
- Creates Azure ML compute machine
- Registers datasets (pandas:1, cats:1, dogs:1)
- Runs ML pipeline with 5 stages:
  ```
  Data Preprocessing (3 parallel jobs) → Data Split → Training → Model Registration
  ```
- Registers final model as `animal-classification:1`
- **Key Code**:
  ```yaml
  az ml job create --file ./pipelines/animals-classification.yaml
  ```
  This submits the pipeline to Azure ML for execution

#### **Job 2: `download` (Download Model)**
- Gets trained model from Azure ML
- Uploads inference code (FastAPI app) as artifact
- **Key Code**:
  ```bash
  az ml model download -n animal-classification -v $VERSION
  ```

#### **Job 3: `deploy` (Deploy to Kubernetes)**
- Builds Docker image with trained model + API
- Pushes to GitHub Container Registry (GHCR)
- Deploys to self-hosted Kubernetes cluster
- Creates 2 replicas for high availability

---

### **2. Azure ML Pipeline** (`pipelines/animals-classification.yaml`)
**Purpose**: Defines ML workflow as a series of components

**Pipeline Structure**:
```
3 Data Prep Jobs (parallel):
├─ data_prep_pandas → preprocesses pandas images
├─ data_prep_cats → preprocesses cat images
└─ data_prep_dogs → preprocesses dog images
    ↓
data_split → combines all and splits into train/test (80/20)
    ↓
training → trains TensorFlow model on training data
    ↓
register → registers model in Azure ML registry
```

**Why this design?**
- **Parallel execution** - 3 datasets processed simultaneously (faster)
- **Reusable components** - Each step is a registered component in Azure
- **Tracking** - Every run is tagged with GitHub SHA for full traceability

---

### **3. FastAPI Inference Server** (`inference/main.py`)
**Purpose**: REST API that serves model predictions

**Key Endpoints**:

#### `POST /predict` - Make predictions
```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Load image
    image = Image.open(BytesIO(contents))
    
    # 2. Preprocess (resize to 224x224, normalize)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    
    # 3. Predict using loaded model
    predictions = model.predict(image_array)
    
    # 4. Return results
    return {
        "prediction": "dogs",
        "confidence": 0.95,
        "all_probabilities": {
            "pandas": 0.02,
            "cats": 0.03,
            "dogs": 0.95
        }
    }
```

**What it does**:
1. Accept image file (JPEG/PNG)
2. Resize to 224x224 pixels
3. Normalize pixel values (0-1)
4. Run through model
5. Return predicted class + confidence scores

#### `GET /health` - Health check (Kubernetes uses this)
```python
@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "unhealthy"}
    return {"status": "healthy", "model_loaded": True}
```

Kubernetes calls this regularly:
- **Liveness probe** - Is the app still running?
- **Readiness probe** - Is it ready to accept requests?
- If fails 3 times → Kubernetes restarts the pod

#### `GET /model-info` - Model metadata
```python
@app.get("/model-info")
async def model_info():
    return {
        "model_name": "animal-classification",
        "input_shape": model.input_shape,
        "classes": ["pandas", "cats", "dogs"],
        "image_size": 224
    }
```

---

### **4. Docker Configuration** (`Dockerfile`)
**Purpose**: Package the API into a container image

```dockerfile
FROM python:3.11-slim              # Start with Python 3.11 image

WORKDIR /app                        # Set working directory

COPY inference/requirements.txt .   # Copy dependencies
RUN pip install --no-cache-dir -r requirements.txt  # Install Python packages

COPY inference/main.py .            # Copy FastAPI app code

EXPOSE 8000                         # Expose port 8000
CMD ["uvicorn", "main:app", ...]   # Run FastAPI server
```

**What it does**:
1. Starts with official Python 3.11 image
2. Installs FastAPI, TensorFlow, etc.
3. Copies application code
4. Runs FastAPI server on port 8000

**Result**: `ghcr.io/danyukezz/mlops-animals-api:main` - A ready-to-run Docker image

---

### **5. Kubernetes Deployment** (`k8s/deployment.yaml`)
**Purpose**: Run Docker image in Kubernetes cluster

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: animals-api
spec:
  replicas: 2                    # Run 2 copies for high availability
  selector:
    matchLabels:
      app: animals-api          # Label selector
  template:
    spec:
      containers:
      - name: animals-api
        image: ghcr.io/danyukezz/mlops-animals-api:main  # Use our Docker image
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"      # Minimum memory needed
          limits:
            memory: "512Mi"      # Maximum memory allowed
        
        # Health checks
        livenessProbe:           # Is it alive?
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        readinessProbe:          # Is it ready?
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**What it does**:
- Runs 2 replicas (pods) of the API
- Each pod has 256-512Mi memory
- Automatically restarts if health check fails
- Kubernetes distributes traffic between replicas

---

### **6. Kubernetes Service** (`k8s/service.yaml`)
**Purpose**: Expose API to outside world

```yaml
apiVersion: v1
kind: Service
metadata:
  name: animals-api
spec:
  type: LoadBalancer              # External access
  ports:
  - port: 80                      # External port
    targetPort: 8000              # Internal port (FastAPI)
  selector:
    app: animals-api              # Route to pods with this label
```

**What it does**:
- Creates external IP address
- Routes port 80 (HTTP) → 8000 (FastAPI)
- Users access: `http://<external-ip>/predict`
- Kubernetes distributes requests between 2 replicas

---

## **How Everything Connects** 🔗

```
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Repository                          │
│  (Code + Workflow + Pipeline Definitions)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ git push main
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          GitHub Actions (azure-ml-workflow.yml)             │
│  ┌─ Job 1: azure-pipeline ─────────────────────────────┐   │
│  │  Runs ML training on Azure ML                        │   │
│  │  - Processes 3 datasets (pandas, cats, dogs)        │   │
│  │  - Trains TensorFlow model                          │   │
│  │  - Registers model as animal-classification:1      │   │
│  └────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─ Job 2: download ────────────────────────────────────┐  │
│  │  Downloads trained model from Azure                  │  │
│  │  Uploads inference code to artifact                  │  │
│  └────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  ┌─ Job 3: deploy ──────────────────────────────────────┐  │
│  │  Builds Docker image                                 │  │
│  │  Pushes to GHCR (ghcr.io/danyukezz/mlops...)        │  │
│  │  Deploys to Kubernetes cluster                       │  │
│  └────────────────────────────────────────────────────┘  │
└───────────────────────┬────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ↓                               ↓
    ┌─────────────┐           ┌──────────────────┐
    │ Azure Cloud │           │  Kubernetes      │
    ├─────────────┤           │  Cluster         │
    │- Datasets   │           ├──────────────────┤
    │- Components │           │ Pod 1: API       │
    │- Model      │           │ Pod 2: API       │
    │- Registry   │           │                  │
    └─────────────┘           │ Service          │
                              │ (port 80)        │
                              └──────────────────┘
                                      ↓
                              ┌─────────────────┐
                              │  External Users │
                              │ POST /predict   │
                              │ GET /health     │
                              │ GET /model-info │
                              └─────────────────┘
```

---

## **The Complete Data Flow** 📊

### **During Training (azure-pipeline job)**
```
Input Data (Local):
├─ pandas images (100 images)
├─ cat images (100 images)
└─ dog images (100 images)

                ↓
        [Upload to Azure Blob]

                ↓
Azure ML Pipeline Execution:
├─ Step 1: Data Prep (3 parallel)
│  ├─ Resize pandas images → 224x224
│  ├─ Resize cat images → 224x224
│  └─ Resize dog images → 224x224
│
├─ Step 2: Data Split
│  ├─ Combine all preprocessed data
│  ├─ Split into train (240 images) and test (60 images)
│
├─ Step 3: Training
│  ├─ Load training data
│  ├─ Train TensorFlow CNN model (5 epochs)
│  ├─ Evaluate on test data
│  └─ Output: trained_model.h5
│
└─ Step 4: Register
   └─ Register model → animal-classification:1

                ↓
        [Model stored in Azure ML]
```

### **During Deployment (deploy job)**
```
Trained Model (from Azure) + Inference Code (from GitHub)
        ↓
    [Docker Build]
    ├─ Install FastAPI, TensorFlow
    ├─ Copy model into container
    ├─ Copy inference code
    └─ Build image
        ↓
    [Push to GHCR]
    └─ Image: ghcr.io/danyukezz/mlops-animals-api:main
        ↓
    [Kubernetes Deploy]
    ├─ Create Deployment (2 replicas)
    ├─ Create Service (LoadBalancer)
    ├─ Each Pod:
    │  ├─ Starts container
    │  ├─ Loads model into memory
    │  ├─ Starts FastAPI server (port 8000)
    │  └─ Passes health checks
    │
    └─ Service routes port 80 → 8000
        ↓
✅ API Ready for predictions
```

### **During Prediction (User Request)**
```
User uploads dog.jpg
        ↓
    POST /predict
        ↓
    [Request reaches Kubernetes Service]
    ├─ LoadBalancer picks Pod 1 or Pod 2
        ↓
    [Inside Pod - FastAPI processes]
    ├─ Load image: dog.jpg
    ├─ Preprocess:
    │  ├─ Convert to RGB
    │  ├─ Resize to 224x224
    │  ├─ Normalize pixel values (0-1)
    │  └─ Add batch dimension
    │
    ├─ Run model prediction:
    │  └─ model.predict(image) → [0.02, 0.03, 0.95]
    │
    └─ Return JSON response:
       {
         "prediction": "dogs",
         "confidence": 0.95,
         "all_probabilities": {
           "pandas": 0.02,
           "cats": 0.03,
           "dogs": 0.95
         }
       }
```

---

## **Key Technologies & Why We Use Them** 🛠️

| Technology | Purpose | Why |
|-----------|---------|-----|
| **GitHub Actions** | CI/CD automation | Trigger workflows automatically on code push |
| **Azure ML** | ML training at scale | Handle large datasets, GPU acceleration, component reusability |
| **TensorFlow** | Deep learning framework | Train CNN for image classification |
| **Docker** | Container packaging | Same code runs everywhere (local, cloud, Kubernetes) |
| **Kubernetes** | Orchestration | Auto-scaling, high availability, load balancing |
| **FastAPI** | Web framework | Simple, fast REST API for inference |

---

## **What We Learned** 📚

1. **MLOps is automation** - Manual training → automatic pipelines
2. **Reproducibility** - Every run tagged with GitHub SHA for traceability
3. **Scalability** - Kubernetes handles multiple requests automatically
4. **Separation of concerns** - Training (Azure) separate from serving (K8s)
5. **CI/CD for ML** - Code → Model → Deployment fully automated

---

## **How to Use the Pipeline** 🚀

### **Trigger the workflow:**
```bash
git push origin main
```

### **Monitor progress:**
GitHub → Actions → Azure ML Workflow

### **Test the API (once deployed):**
```bash
# Get external IP
kubectl get svc animals-api

# Test prediction
curl -X POST http://<external-ip>/predict \
  -F "file=@dog.jpg"

# Test health
curl http://<external-ip>/health
```

### **Scale to more replicas:**
Edit `k8s/deployment.yaml` → Change `replicas: 2` to `replicas: 5`

```bash
kubectl apply -f k8s/deployment.yaml
# Kubernetes automatically creates 3 more pods
```

---

## **Summary** ✅

**What is it?**
- Automated ML pipeline from code to production

**What does it do?**
- Trains animal classifier on Azure ML
- Packages into Docker image
- Deploys to Kubernetes for serving

**How?**
- GitHub Actions triggers on code push
- 3 sequential jobs: train → download → deploy
- Final result: scalable REST API ready for predictions

**Why?**
- **Speed**: Auto-triggered, no manual steps
- **Reliability**: Health checks, auto-restart
- **Scalability**: Kubernetes handles load
- **Traceability**: Every run tagged with code version

---

## **File Structure Reference**

```
Labo5/
├── .github/workflows/
│   └── azure-ml-workflow.yml          # Main workflow
├── pipelines/
│   └── animals-classification.yaml    # ML pipeline definition
├── inference/
│   ├── main.py                        # FastAPI app
│   └── requirements.txt               # Python dependencies
├── k8s/
│   ├── deployment.yaml                # Kubernetes deployment
│   └── service.yaml                   # Kubernetes service
├── Dockerfile                         # Docker image spec
└── README.md                          # Project info
```

---

**🎉 You've built a production-ready MLOps system!**


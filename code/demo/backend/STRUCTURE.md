# Backend Structure Documentation

## Overview
The backend has been reorganized into logical sub-folders for better maintainability and clarity.

## Directory Structure

```
backend/
├── api/                    # API endpoints and routes
│   ├── __init__.py
│   ├── routes.py          # Main API routes
│   ├── database_routes.py # Database-specific routes
│   ├── sample_queries_routes.py
│   └── sample_query_routes.py
│
├── core/                   # Core application components
│   ├── __init__.py
│   ├── main.py            # Main FastAPI application
│   ├── production_main.py # Production-ready main
│   ├── start_server.py    # Server startup logic
│   ├── shared_models.py   # Shared Pydantic models
│   ├── api_server.py      # API server implementation
│   └── config_server.py   # Configuration server
│
├── models/                 # ML models and pipelines
│   ├── __init__.py
│   ├── pipelines.py       # Pipeline implementations
│   ├── model_config.py    # Model configurations
│   ├── fixed_phobert_sql.py
│   └── sql_templates.py   # SQL template handling
│
├── database/              # Database management
│   ├── __init__.py
│   ├── db_manager_normalized.py
│   ├── migration_script.py
│   ├── csv_to_normalized_importer.py
│   ├── normalized_schema.sql
│   ├── complex_join_queries.sql
│   └── mermaid_schema.md
│
├── evaluation/            # Model evaluation and metrics
│   ├── __init__.py
│   ├── local_evaluation.py
│   ├── local_model_evaluator.py
│   ├── real_model_evaluator.py
│   ├── simple_local_evaluator.py
│   └── improved_evaluation_metrics.py
│
├── analysis/              # Statistical analysis and reporting
│   ├── __init__.py
│   ├── statistical_analysis.py
│   ├── statistical_evaluation.py
│   ├── statistical_visualizations.py
│   ├── comprehensive_analysis_generator.py
│   ├── metrics_dashboard_generator.py
│   ├── demo_statistical_framework.py
│   └── api_execution_analyzer.py
│
├── utils/                 # Utility functions and helpers
│   ├── __init__.py
│   ├── colab_client.py    # Colab API client
│   ├── sample_queries.py  # Sample query generation
│   ├── sample_queries_data.py
│   ├── vietnamese_test_queries.py
│   ├── ground_truth_generator.py
│   └── import_tiki_data.py
│
├── scripts/               # Automation scripts and fixes
│   ├── __init__.py
│   ├── fix_analysis_structure.py
│   ├── fix_pipeline2_response.py
│   ├── quick_fix_json.py
│   ├── analyze_unused_files.py
│   ├── run_evaluation.sh
│   └── start_api.sh
│
├── tests/                 # Test files
│   ├── __init__.py
│   ├── test_colab_connection.py
│   ├── test_fixed_pipeline1.py
│   └── test_statistical_framework.py
│
├── config/                # Configuration files
│   └── config.py
│
├── data/                  # Data storage (empty)
├── local_models/          # Local model storage (empty)
├── batch_results/         # Batch execution results
│   └── *.json
│
├── requirements.txt       # Python dependencies
├── README.md             # Main documentation
├── STRUCTURE.md          # This file
└── backend.log           # Application logs
```

## Component Descriptions

### 🔧 Core (`core/`)
Contains the main application logic, server startup, and shared models.
- **Entry Points**: `main.py`, `production_main.py`
- **Server Logic**: `start_server.py`, `api_server.py`
- **Shared Components**: `shared_models.py`

### 🌐 API (`api/`)
RESTful API endpoints for the Vietnamese NL2SQL system.
- **Main Routes**: Search, pipeline execution, health checks
- **Database Routes**: Database operations and queries
- **Sample Query Routes**: Test query management

### 🤖 Models (`models/`)
Machine learning models and pipeline implementations.
- **Pipeline Logic**: Vietnamese→SQL translation pipelines
- **Model Configuration**: Model loading and setup
- **SQL Templates**: Template-based SQL generation

### 🗄️ Database (`database/`)
Database schema, migrations, and management utilities.
- **Schema**: Normalized e-commerce database structure
- **Migrations**: Database version control
- **Import Tools**: Data loading utilities

### 📊 Evaluation (`evaluation/`)
Model performance evaluation and metrics calculation.
- **Local Evaluation**: Offline model testing
- **Metrics**: EX, EM, latency, accuracy calculations
- **Comparative Analysis**: Multi-pipeline comparison

### 📈 Analysis (`analysis/`)
Statistical analysis, reporting, and visualization.
- **Statistical Framework**: Comprehensive analysis tools
- **Visualizations**: Charts, graphs, dashboards
- **Report Generation**: Automated reporting

### 🛠️ Utils (`utils/`)
Utility functions, helpers, and external integrations.
- **Colab Integration**: API client for Colab services
- **Data Generation**: Sample query creation
- **Import Tools**: Data processing utilities

### 📜 Scripts (`scripts/`)
Automation scripts, fixes, and maintenance tools.
- **Deployment Scripts**: Server startup automation
- **Fix Scripts**: Data and structure repairs
- **Analysis Tools**: Code and usage analysis

### 🧪 Tests (`tests/`)
Test suites for various components.
- **Integration Tests**: API and pipeline testing
- **Unit Tests**: Component-specific testing
- **Connection Tests**: External service validation

## Usage Examples

### Starting the Server
```bash
# Development
python core/main.py

# Production
python core/production_main.py

# Using script
./scripts/start_api.sh
```

### Running Evaluations
```bash
# Local evaluation
python evaluation/local_evaluation.py

# Statistical analysis
python analysis/statistical_analysis.py

# Using script
./scripts/run_evaluation.sh
```

### Testing
```bash
# Test Colab connection
python tests/test_colab_connection.py

# Test Pipeline 1
python tests/test_fixed_pipeline1.py
```

## Import Path Updates

Due to the reorganization, import paths need to be updated:

```python
# Old
from shared_models import SearchRequest
from pipelines import pipeline1

# New
from core.shared_models import SearchRequest
from models.pipelines import pipeline1
```

## Benefits of New Structure

1. **🎯 Clear Separation of Concerns**: Each folder has a specific purpose
2. **📦 Better Modularity**: Components are logically grouped
3. **🔍 Easier Navigation**: Developers can quickly find relevant files
4. **🧪 Improved Testing**: Test files are separated from implementation
5. **📚 Better Documentation**: Structure is self-documenting
6. **🚀 Scalability**: Easy to add new components in appropriate folders

## Migration Notes

- All files have been moved to appropriate sub-folders
- Python package structure maintained with `__init__.py` files
- Import paths need to be updated in existing code
- Scripts and automation tools are now centralized
- Configuration and data folders are clearly separated

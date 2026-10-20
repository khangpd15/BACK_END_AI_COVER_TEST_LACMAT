# RemiCare Cover Test AI Backend

FastAPI backend độc lập chuyên phục vụ sàng lọc Cover Test (webcam-based Cover Test screening).

## Chức năng
- Tiếp nhận và xác thực chuỗi dữ liệu thời gian chuyển động mắt (ocular landmarks time-series).
- Lưu trữ session cục bộ tại `data/cover_test/sessions/<session_id>/` (metadata.json, sampling.json).
- Lưu trữ session đám mây vào PostgreSQL (Supabase) và Supabase Storage (`cover-test-raw`).
- Trích xuất đặc trưng chuyển động theo chu kỳ (Cover Cycle Feature Extraction) và đồng thuận thời gian (Temporal Consensus).
- Tuyệt đối bảo vệ quyền riêng tư: Không lưu trữ hình ảnh sinh trắc học của khách hàng.

## Cấu trúc thư mục

```text
backend_cover_test/
├── app/
│   ├── main.py                 # FastAPI application & lifespan
│   ├── config.py               # Cấu hình môi trường Cover Test
│   ├── api/
│   │   ├── cover_test.py       # POST /api/v1/cover-test/sessions
│   │   └── cover_test_session.py # POST /api/cover-test/sessions
│   ├── schemas/                # Schemas Cover Test & Screening
│   ├── services/
│   │   ├── cover_test/         # SessionService & SupabaseStorageService
│   │   ├── cv/                 # Quality gate & Temporal aggregator
│   │   ├── fps_model_service.py # Numeric trajectory consensus service
│   │   ├── validation.py       # Time-series validation
│   │   └── shared_feature_contract.py # 30 shared features contract
│   └── db/
│       ├── database.py         # SQLAlchemy Async Engine & auto migrations
│       ├── models/             # cover_test_session, cover_test_cycle, cover_test_result
│       └── repositories/       # cover_test_repository
├── data/
│   └── cover_test/sessions/    # Session local do API tạo
├── models/
│   └── candidates/             # Chứa artifact model candidate
├── tests/                      # Bộ kiểm thử Cover Test
├── requirements.txt            # Thư viện phụ thuộc
├── render.yaml                 # Cấu hình deploy Render
└── .env.example                # Mẫu biến môi trường
```

## Cài đặt và chạy local

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Swagger API Docs: `http://127.0.0.1:8001/docs`

## Deploy lên Render
- Service Type: Web Service
- Name: `remicare-cover-test-ai`
- Runtime: Python 3
- Build Command: `python -m pip install -r requirements.txt`
- Start Command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
- Health Check Path: `/ping`

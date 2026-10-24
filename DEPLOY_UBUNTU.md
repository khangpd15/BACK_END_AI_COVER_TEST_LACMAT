# Deploy hai backend trên VPS Ubuntu bằng Docker

Hai backend đúng là:

- Cover Test: `D:\REMICARE-STRABISMUS-AI-COVER-TEST\backend_cover_test`
- Hirschberg: `D:\REMICARE-STRABISMUS-AI-HIRSCHBERG`

Không cần copy thư mục train, ảnh bệnh nhân, video hoặc virtualenv lên VPS.
Hirschberg có đủ bundle runtime trong `app/hirschberg_runtime` (40 file đã kiểm tra hash).
Dockerfile dùng CPU, Python 3.12 và một worker mỗi backend. Cấu hình này nhắm tới
Ubuntu x86_64/amd64; không chọn VPS ARM cho bộ wheel MediaPipe hiện tại.
Gợi ý điểm bắt đầu để đo tải: 2 vCPU, 4 GB RAM, 20 GB đĩa trống. Đây không phải
cam kết sức tải; cần đo RAM/latency với dữ liệu thực và số người dùng đồng thời.

## 1. Cài Docker trên Ubuntu

Cài Docker Engine, Buildx và Compose plugin bằng apt repository theo
[hướng dẫn chính thức](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).
Kiểm tra:

```bash
sudo docker version
sudo docker compose version
uname -m
```

`uname -m` cần là `x86_64`. Dùng `sudo docker ...` trong các lệnh dưới nếu tài khoản
chưa có quyền Docker.

## 2. Đưa hai backend lên VPS

Sắp thư mục:

```text
/opt/remicare/
  cover-test/     # nội dung backend_cover_test
  hirschberg/    # nội dung REMICARE-STRABISMUS-AI-HIRSCHBERG
```

Có thể đổi tên thư mục bằng cách sửa `HIRSCHBERG_CONTEXT` trong `.env.vps`.
Các file bắt buộc:

- Cover: `app/`, `models/`, `docker/`, `Dockerfile`, `.dockerignore`,
  `requirements-docker.txt`, `compose.yaml`, `compose.vps.yaml`, `Caddyfile`,
  `.env.docker.example`, `.env.vps.example`.
- Hirschberg: `app/` (đặc biệt toàn bộ `app/hirschberg_runtime/`), `scripts/verify_runtime_bundle.py`,
  `Dockerfile`, `.dockerignore`, `requirements.txt`, `requirements-render.txt`,
  `compose.yaml`, `.env.docker.example`.

Không upload `.venv*`, `venv`, `.env` đang dùng trên Windows, dữ liệu huấn luyện,
video hoặc báo cáo chứa ảnh. `.dockerignore` đã giới hạn build context nhưng
không thay thế việc chọn đúng file khi upload. Nếu dùng Git LFS, lấy đủ file
checkpoint thật: build sẽ kiểm tra hash và từ chối LFS pointer.

## 3. Cấu hình

```bash
cd /opt/remicare/cover-test
cp .env.docker.example .env.docker
cp .env.vps.example .env.vps
chmod 600 .env.docker .env.vps
nano .env.docker
nano .env.vps
```

Thay mọi `CHANGE_ME`:

- `.env.docker`: URL PostgreSQL/Supabase, URL Storage, service-role key và origin
  frontend thật. Password trong DATABASE_URL phải URL-encode ký tự đặc biệt.
  Dùng Supabase pooler phù hợp nếu VPS không có IPv6. Key chỉ nằm ở backend.
- Tạo bucket **private** `cover-test-raw` trong Supabase hoặc điền tên bucket hiện có.
  Container không tự tạo bucket. Database user cần quyền tạo bảng/index lần đầu.
- `.env.vps`: hai hostname API khác nhau, email cấp chứng chỉ và origin frontend.
  Origin frontend ở hai file phải trùng nhau, có `https://`, không có dấu `/` cuối.

Ví dụ hostname: `cover-api.tenmien.vn` và `eye-api.tenmien.vn`.
Trỏ DNS A của cả hai về IP VPS; chỉ thêm AAAA khi IPv6 thực sự hoạt động.
Mở TCP 80/443 ở firewall nhà cung cấp/VPS. Caddy tự cấp/gia hạn HTTPS
([điều kiện hoạt động](https://caddyserver.com/docs/automatic-https)).
Không mở cổng 8000/8001 ra Internet. Nếu máy đang chạy Nginx/Caddy khác trên 80/443,
dùng chế độ từng backend ở phần 6 thay vì chạy thêm proxy này.

Cover Test container sẽ từ chối khởi động khi thiếu PostgreSQL/Storage/origin,
hoặc Storage client không khởi tạo được; không dùng fallback lưu RAM trong cấu hình này.
Healthcheck kiểm tra DB, chưa xác minh quyền ghi bucket. Kiểm tra một phiên upload
thực sau deploy để xác nhận cả JSON và JPEG đều vào bucket.

## 4. Build và chạy chung

```bash
cd /opt/remicare/cover-test
docker compose --env-file .env.vps -f compose.vps.yaml config -q
docker compose --env-file .env.vps -f compose.vps.yaml build --pull
docker compose --env-file .env.vps -f compose.vps.yaml up -d
docker compose --env-file .env.vps -f compose.vps.yaml ps
docker compose --env-file .env.vps -f compose.vps.yaml logs --tail=100
```

Build Hirschberg chạy `pip check`, kiểm tra import native và hash runtime.
Đợi model nạp; `/ready` trả 503 trong lúc nạp hoặc khi lỗi. Sau đó:

```bash
curl -f https://cover-api.tenmien.vn/ping
curl -f https://cover-api.tenmien.vn/health/db
curl -f https://eye-api.tenmien.vn/ready
curl -f https://eye-api.tenmien.vn/api/v1/hirschberg/health
```

`/health/db` cần `"connected": true` (endpoint vẫn có thể HTTP 200 khi DB lỗi;
Docker healthcheck có kiểm tra nội dung này). API Cover vẫn có thể hoạt động khi
chưa có model fusion và trả INCONCLUSIVE; trạng thái container healthy không có
nghĩa mô hình kết hợp đã được huấn luyện.

Frontend:

```dotenv
VITE_AI_BACKEND_URL=https://cover-api.tenmien.vn
VITE_HIRSCHBERG_BACKEND_URL=https://eye-api.tenmien.vn
```

Build/deploy lại frontend sau khi đổi biến Vite. URL không thêm `/api/...`.

## 5. Mô hình và dữ liệu

- Hirschberg chạy bundle production sẵn có, không tải pretrained weights lúc chạy.
- Cover nhận/lưu timestamp và crop bỏ che. Docker image đóng gói artifact
  `models/cover_multimodal_model.joblib` và Compose trỏ
  `COVER_MULTIMODAL_MODEL_PATH=/srv/remicare/models/cover_multimodal_model.joblib`.
  Health endpoint trả trạng thái riêng trong `multimodalModel`.
- Hai API không tự trở thành một model fusion khi đặt chung VPS.
  Nhánh nghiên cứu v0.5 dùng subprocess từ thư mục train trên Windows **không được
  đóng gói vào image Cover**. Không đặt đường dẫn `D:\...` trong container; cấu hình
  mặc định vẫn lưu hai loại dữ liệu và báo thiếu model kết hợp như trước.
- Timestamps/ảnh Cover nằm ở PostgreSQL/Supabase; volume `cover-data` dành cho dữ liệu
  local của API cũ. Session Hirschberg dùng volume `hirschberg-sessions` khi yêu cầu
  lưu phiên. Chứng chỉ Caddy nằm ở `caddy-data` và cấu hình ở `caddy-config`.
- Đã tắt cleanup schema cũ mặc định. Không bật `ENABLE_LEGACY_SCHEMA_CLEANUP`
  nếu chưa chủ động quyết định xóa bảng/cột ảnh cũ.

## 6. Chạy từng backend với reverse proxy sẵn có

Cover:

```bash
cd /opt/remicare/cover-test
# Tạo .env.docker theo phần 3 trước.
docker compose up -d --build
curl -f http://127.0.0.1:8000/ping
```

Hirschberg:

```bash
cd /opt/remicare/hirschberg
cp .env.docker.example .env.docker
chmod 600 .env.docker
nano .env.docker
docker compose up -d --build
curl -f http://127.0.0.1:8001/ready
```

Reverse proxy chạy trên host chuyển hostname tương ứng tới 127.0.0.1:8000/8001.
Nếu proxy cũng ở container, dùng stack chung hoặc nối Docker network thích hợp;
127.0.0.1 trong proxy container không phải host.
Chọn **một** cách chạy, không chạy stack chung và các compose riêng cùng lúc.

## 7. Cập nhật, kiểm tra và backup

```bash
cd /opt/remicare/cover-test
docker compose --env-file .env.vps -f compose.vps.yaml up -d --build
docker compose --env-file .env.vps -f compose.vps.yaml logs --tail=100 cover-test hirschberg
docker stats --no-stream
```

`up -d --build` giữ named volumes. Không dùng `down -v` khi cần giữ dữ liệu/chứng chỉ.
Sao lưu PostgreSQL, bucket Supabase và named volumes trước khi nâng cấp; giữ bản
source/image đã chạy tốt để rollback. Nếu Docker báo unhealthy, xem logs:
Docker restart policy chỉ tự restart khi tiến trình thoát, không tự restart chỉ vì unhealthy.

## Phạm vi đã kiểm tra trên máy phát triển

Compose parser đã kiểm tra cả hai cấu hình riêng và cấu hình chung; bundle Hirschberg
40 file đã xác minh hash. Có kiểm thử cấu hình startup và việc không chạy DROP mặc định.
Docker Engine trên máy phát triển chưa chạy, nên chưa build image hoặc xác nhận
native imports/healthcheck trong Linux container. Bước build và smoke test VPS ở
trên vẫn cần thực hiện; chưa deploy lên VPS nào.

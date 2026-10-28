# Cải tiến train BE: crop mắt, timestamp và kiểm định 30 fps

Ngày thực hiện: 28/10/2026. Mã BE: `D:\REMICARE-STRABISMUS-AI-COVER-TEST`.

Đã huấn luyện một mô hình **ảnh thử nghiệm thực sự từ pixel của hai vùng mắt**. Accuracy kiểm tra chéo trên ảnh Hirschberg đạt **94,94%**, nhưng **chưa chứng minh accuracy Cover Test trên 90%**. Balanced accuracy còn 89,92%, specificity 82,14%; định danh người tham gia của bộ ảnh vẫn chờ QC. Không thay các con số này bằng confidence của một lần dự đoán.

## Dữ liệu đã kiểm tra và train

Đường dẫn người dùng cung cấp `D:\REMICARE-STRABISMUS-AI-HIRSCHBERG` là BE chứa runtime, trọng số và hồ sơ xác nhận nguồn. Manifest runtime tham chiếu kho ảnh thực tế tại `D:\REMICARE_TRAIN_AI\HIRSCHBERG_METHOD`.

Nguồn train: `hirschberg_review_all_v1/reviewed_candidates.jsonl`, SHA-256 `236113d3de4d7ea2cea8a6cf83423db53766c03f4811a82537161eaa2a026711`.

- 163 bản ghi review; 161 có `clinical_confirmed=true` theo reviewer.
- 158 ảnh được train: 28 NORMAL, 130 STRABISMUS; loại 2 nhãn chưa xác nhận và 3 ROI thiếu/không hợp lệ.
- Kiểm tra SHA-256 ảnh gốc hoặc bản sao byte-identical trước khi cắt. Không dùng preview hoặc tọa độ pupil/reflex đã gán nhãn làm đặc trưng dự đoán.
- 158 mã subject được khai báo cho 158 ảnh. Cả 163 bản ghi gốc có `training_eligible=false`, trạng thái chờ kiểm tra định danh. Lần chạy này chỉ tạo **research candidate**, giữ nguyên hồ sơ nguồn và không duyệt mô hình chính thức.
- HOG trên từng crop RGB chuyển grayscale 64×32, giữ thứ tự mắt giải phẫu; StandardScaler và LogisticRegression balanced, C=0,01 cố định. Mô hình nhỏ này không cần GPU.
- Kiểm tra chéo 5 fold theo **mã subject được khai báo**, scaler chỉ fit trên train fold. Không coi các mã này là bằng chứng đã xác minh độc lập giữa người tham gia; chưa có tập bệnh nhân mới được khóa riêng.

| Chỉ số kiểm tra chéo ảnh | Kết quả |
|---|---:|
| Accuracy | 94,94% = 150/158 |
| Balanced accuracy | 89,92% |
| Sensitivity STRABISMUS | 97,69% = 127/130 |
| Specificity NORMAL | 82,14% = 23/28 |
| Sai dương / sai âm | 5 / 3 |
| Wilson 95% CI accuracy theo ảnh | 90,33–97,41% |

CI trên tính theo ảnh, không phải CI theo bệnh nhân đã xác minh. Mất cân bằng lớp làm accuracy cao hơn balanced accuracy. Các số này là đánh giá nghiên cứu trên miền ảnh Hirschberg đã review, không phải hiệu năng Cover Test hay hiệu năng thực địa iPhone.

Trọng số: `backend-cover-v3/image-transfer/static_image_transfer_candidate.joblib`. Kết quả đầy đủ: `backend-cover-v3/image-transfer/image_transfer_report.json`.

## Thay đổi FE → BE

1. FE xuất thêm `coordinateSpace=CANTHAL_LOCAL_V1`, `leftEyeLocal`, `rightEyeLocal`, lấy đúng tọa độ theo khóe mắt đã dùng trong đo sự kiện. Mắt bị che/chớp hoặc khung không đo được giữ `valid=false`, tọa độ null. Không thay tọa độ gốc của các mô hình cũ.
2. Chuẩn hóa `timestampMs` nguồn; hiệu thời gian chia 1000 khi tính chuyển động theo giây. `t` tương đối của FE là ms; dữ liệu legacy Korean chỉ có `t` được hiểu theo hợp đồng cũ là giây. Loại cách đoán đơn vị theo độ lớn timestamp. Không nối các chu kỳ reset clock để lấy đạo hàm.
3. Giữ frame ID và timestamp của lúc chụp, không dùng thời điểm JPEG encode/upload hoàn tất. Kiểm tra crop đúng một source frame, kích thước và rectangle trong ảnh nguồn, ảnh không mirror, khoảng bất định onset ≤100 ms. Chuỗi thời gian phải tăng thực sự, không trùng frame.
4. Nhánh v3 dùng chung extractor giữa train và inference: HOG hai ảnh + 12 đặc trưng chuyển động theo khóe mắt. Không trộn tọa độ iris toàn ảnh với tọa độ cục bộ.
5. Cửa sổ sự kiện 0–1.500 ms: ≥15 mẫu hai mắt hợp lệ, bao phủ ≥1 giây; cửa sổ đầu 500 ms có ≥5 mẫu trải ≥300 ms. Khoảng mất tín hiệu muộn có thể bỏ qua nếu còn đuôi liên tục ≥5 mẫu trải ≥200 ms. Không tính vận tốc qua gap >100 ms, không nội suy iris bị che.
6. Mô hình paired chỉ nhận thủ tục cover/uncover đã xác minh. Trước khi đưa ra kết luận cần ≥4 sự kiện trong ≥2 chu kỳ và một artifact đã qua kiểm định độc lập.

Đã sửa lỗi mô hình cũ: có nhận ảnh nhưng chỉ lặp lại xác suất từ quỹ đạo rồi ghi đã phân tích ảnh. Bây giờ nhánh này chỉ khai báo `time_series`, `analyzedEventIds=[]`. Fusion chỉ nhận các event thực sự được nhánh ảnh xử lý, không gán toàn bộ event đã nhận là đã phân tích. Các nhánh cũ giữ `isReliable=false` và `accuracy=null` khi chưa có bằng chứng kiểm định paired.

BE tự nạp trọng số ảnh nghiên cứu tại `models/cover_image_transfer_candidate.joblib`, hoặc đường dẫn `COVER_IMAGE_TRANSFER_MODEL_PATH`. Điểm ảnh nằm trong `imageTransferResearch`, có danh sách event đã xử lý. Chưa dùng trọng số này để tự thay đổi trọng số fusion hoặc kết luận lâm sàng.

## Kiểm tra thiết bị 30 fps

Không tin riêng `samplingRateHz=30` hay tên file. BE tính median fps, fps hiệu dụng trên toàn khoảng thời gian, p95 và gap lớn nhất từ **các source frame được xử lý/gửi lên**, kể cả mẫu mắt không hợp lệ còn được ghi nhận.

Policy kỹ thuật: median và effective fps ≥29,5, p95 interval ≤50 ms, max interval ≤100 ms; chấp nhận mã hóa nominal 29,97/30 fps. Mẫu 15/24 fps bị loại cho nhánh mới, không được nâng thành 30 fps bằng nội suy. Đây là điều kiện chất lượng kỹ thuật ban đầu, chưa phải ngưỡng thiết bị đã kiểm định lâm sàng. Camera 30/60 fps nhưng FE chỉ xử lý 15 fps sẽ không qua.

| Video iPhone | Frames | Effective fps offline | Số PNG crop | Sự kiện uncover có crop |
|---|---:|---:|---:|---:|
| 4K_30FPS | 2.105 | 29,9573 | 56 | 6 |
| 4K_60FPS | 2.094 | 30,0000 | 60 | 6 |
| HD_30FPS | 2.116 | 29,9575 | 56 | 5 |

Cả ba qua policy **nhịp PTS nguồn offline**; tên 4K/60 không chứng minh nguồn thực sự 4K/60. Đây không phải benchmark throughput trên trình duyệt iPhone.

Extractor chuyển động v3 chạy được 5/6, 6/6, 3/5 cửa sổ tương ứng; ba cửa sổ còn lại thiếu đuôi settling liên tục. Việc qua extractor không đồng nghĩa đủ điều kiện train. Hướng mirror của video gốc vẫn UNKNOWN; kiểm tra v3 không tự biến UNKNOWN thành false. 17 dòng manifest paired đều chưa đủ điều kiện train vì thiếu nhãn lâm sàng liên kết video, subject ID xác minh, phân chia train/validation/test và nguồn thiết bị/cơ sở đầy đủ.

Ba kết quả chi tiết:

- `backend-cover-v3/4K_30FPS_backend_audit.json`
- `backend-cover-v3/4K_60FPS_backend_audit.json`
- `backend-cover-v3/HD_30FPS_backend_audit.json`

## Pipeline train paired và điều kiện vượt 90%

`scripts/cover_training_v3/train.py` audit mặc định; `--train` chỉ train khi đủ dữ liệu. Mỗi dòng cần nhãn bác sĩ, ID người tham gia xác minh, session/source SHA, thiết bị/cơ sở, split cố định, crop, event, sample và bằng chứng thủ tục. JSON mẫu đọc được nằm tại `backend-cover-v3/iphone_training_manifest.json`; các giá trị chưa biết cố ý để null.

- Cấm một subject, video hoặc ảnh trùng SHA xuất hiện qua các nhóm/split khác nhau.
- Train/validation/test tách theo subject. Tối thiểu mỗi lớp: 5 subject train, 3 validation, 20 test; đây là sàn kỹ thuật, không bảo đảm đủ cỡ mẫu khoa học.
- Chọn C bằng validation; chỉ chạy holdout sau chọn mô hình. Không refit trên holdout, không tự tìm nhiều random seed để lấy con số đẹp.
- Đánh giá theo subject, kiểm tra từng session có ≥4 sự kiện trong ≥2 chu kỳ. Session không đủ hoặc bị từ chối được báo coverage và tính là không đạt trong all-attempt accuracy.
- Chỉ đánh dấu đủ điều kiện promotion khi accuracy, balanced accuracy, sensitivity và specificity **đều >90%**, cận dưới Wilson 95% của accuracy >90%, coverage=100%. Kết quả chỉ có phạm vi quần thể/thiết bị/cơ sở đã kiểm tra; cần đánh giá thêm theo thiết bị trước khi công bố một mẫu máy phù hợp.
- Artifact paired được cấu hình riêng qua `COVER_PAIRED_MODEL_PATH`. Nếu cấu hình sai hoặc thiếu thì trả INCONCLUSIVE, không tự fallback để nhận vơ độ chính xác đã kiểm định.

Lần chạy `--train` với ba iPhone kết thúc đúng trạng thái `NEEDS_VERIFIED_PAIRED_SUBJECTS`, không tạo mô hình từ nhãn đoán. Báo cáo: `backend-cover-v3/paired-training/training_report.json`.

Lệnh tái lập từ `D:\AI_Check_Lac`:

```powershell
python scripts/cover_training_v3/train_image_transfer.py --root D:/REMICARE_TRAIN_AI/HIRSCHBERG_METHOD --output docs/reports/backend-cover-v3/image-transfer
python scripts/cover_training_v3/audit_iphone.py
python scripts/cover_training_v3/train.py --manifest docs/reports/backend-cover-v3/iphone_training_manifest.json --output docs/reports/backend-cover-v3/paired-training --train
```

Tài liệu phương pháp: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [tránh rò rỉ dữ liệu và fit preprocessing đúng tập train](https://scikit-learn.org/stable/common_pitfalls.html).

## Kiểm chứng và phạm vi áp dụng

Đã chạy 32 test BE về inference, liên kết crop/frame, ms/giây, cadence, gap, eye masking và artifact ảnh; 5 test train về rò rỉ/nhãn/CI; 109 test Node hiện đọc được; FE build thành công. Hai test cũ dùng luồng async/API bị treo trong môi trường hiện tại nên không tính là passed. File test FE cũ `scripts/uncoverCapture.test.js` vẫn chứa NUL từ trước, được giữ nguyên và loại khỏi lần chạy Node. Chưa kiểm thử trên Safari iPhone hoặc triển khai lên Render.

Cần bổ sung để hoàn tất mục tiêu accuracy Cover Test >90%: video có lác và NORMAL với nhãn khám độc lập, ID người tham gia đã xác minh, metadata thiết bị/hướng ảnh và tập người mới được khóa trước khi train. Ảnh Hirschberg hiện có hỗ trợ nhánh hình thái mắt nhưng không cung cấp phản ứng mắt theo thời gian thay cho video này.

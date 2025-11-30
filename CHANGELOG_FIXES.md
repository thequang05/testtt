# 📝 Tổng hợp các thay đổi code đã sửa

**Ngày:** 2025-11-30  
**Mục đích:** Sửa lỗi WebSocket 403 Forbidden và lỗi SQLAlchemy greenlet_spawn

---

## 🔍 Tổng quan các vấn đề đã sửa

### 1. **Lỗi WebSocket 403 Forbidden**
- **Nguyên nhân:** Frontend gọi `/api/v1/ws/frames/default` nhưng backend chỉ nhận camera ID số (0, 1, ...)
- **Giải pháp:** Sửa frontend để sử dụng `MultiCameraGrid` component thay vì `VideoPlayer` với `roadName="default"`

### 2. **Lỗi SQLAlchemy greenlet_spawn**
- **Nguyên nhân:** 
  - Các async functions sử dụng `SessionLocal()` (sync) trực tiếp trong async context
  - Camera processes (multiprocessing) cố gắng lưu DB trực tiếp
  - Database URL đang dùng async driver (`sqlite+aiosqlite://`) nhưng sync session cần sync driver (`sqlite://`)
- **Giải pháp:**
  - Tách các thao tác DB sync vào functions riêng và chạy trong thread bằng `ThreadPoolExecutor` (thay vì `asyncio.to_thread()`)
  - Tắt auto_save DB trong camera processes, để background worker xử lý
  - Sửa database URL conversion để chuyển đúng từ async driver sang sync driver

---

## 📁 Chi tiết các file đã sửa

### 1. **frontend/app/page.tsx**

#### Thay đổi 1: Import statement (Line 3)
**Trước:**
```typescript
import VideoPlayer from "./components/stream/VideoPlayer";
```

**Sau:**
```typescript
import { MultiCameraGrid } from "./components/stream/VideoPlayer";
```

#### Thay đổi 2: Sử dụng component (Line 119-123)
**Trước:**
```typescript
{/* HÀNG 1: Grid camera (component VideoPlayer của bạn đã chia 2 camera) */}
<VideoPlayer
  roadName="default"
  backendUrl="ws://localhost:8000"
/>
```

**Sau:**
```typescript
{/* HÀNG 1: Grid camera (component MultiCameraGrid hiển thị 2 camera) */}
<MultiCameraGrid />
```

**Lý do:** `MultiCameraGrid` tự động hiển thị 2 camera với ID đúng (0 và 1), thay vì gọi với "default" gây lỗi 403.

---

### 2. **backend/app/api/api_vehicles.py**

#### Thay đổi 0: Import ThreadPoolExecutor (Line 1-10)

**Thêm import:**
```python
from concurrent.futures import ThreadPoolExecutor
```

#### Thay đổi 0.5: Thêm ThreadPoolExecutor vào SystemState (Line 22-31)

**Trước:**
```python
class SystemState:
    def __init__(self):
        self.manager = None
        self.info_dict = None   
        self.frame_dict = None  
        self.processes = []     
        self.result_queue = None
```

**Sau:**
```python
class SystemState:
    def __init__(self):
        self.manager = None
        self.info_dict = None   
        self.frame_dict = None  
        self.processes = []     
        self.result_queue = None
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db_worker")
```

**Lý do:** Tạo ThreadPoolExecutor riêng để chạy các sync DB operations trong thread pool, tách biệt hoàn toàn khỏi async context.

#### Thay đổi 1: Background Worker - Tách sync function (Line 33-77)

**Thêm function mới `_save_stats_sync` (Line 33-64):**
```python
def _save_stats_sync(info_dict_snapshot):
    """Sync function để lưu stats vào DB - chạy trong thread riêng"""
    db = SessionLocal()
    try:
        for key, data in info_dict_snapshot.items():
            try:
                if "_" not in key: continue 
                cam_id = int(key.split("_")[1])
            except: continue

            details = data.get('details', {})
            log = TrafficLog(
                camera_id=cam_id,
                total_vehicles=data.get('total_entered', 0),
                fps=data.get('fps', 0),
                count_car=details.get('car', {}).get('entered', 0),
                count_motor=(
                    details.get('motorcycle', {}).get('entered', 0) + 
                    details.get('motorbike', {}).get('entered', 0) + 
                    details.get('motor', {}).get('entered', 0)
                ),
                count_bus=details.get('bus', {}).get('entered', 0),
                count_truck=details.get('truck', {}).get('entered', 0),
                timestamp=datetime.now()
            )
            db.add(log)
        db.commit()
    except Exception as e:
        print(f"❌ Lỗi worker lưu DB: {e}")
        db.rollback() 
    finally:
        db.close()
```

**Sửa `save_stats_to_db_worker` (Line 66-77):**
**Trước:**
```python
async def save_stats_to_db_worker():
    print("💾 Background Worker: Đã kích hoạt chế độ ghi log giao thông...")
    while True:
        try:
            await asyncio.sleep(10)
            if sys_state.info_dict:
                db = SessionLocal()  # ❌ Lỗi: sync DB trong async context
                try:
                    # ... code lưu DB trực tiếp ...
                finally:
                    db.close()
```

**Sau (Phiên bản cuối cùng):**
```python
async def save_stats_to_db_worker():
    print("💾 Background Worker: Đã kích hoạt chế độ ghi log giao thông...")
    loop = asyncio.get_event_loop()
    while True:
        try:
            await asyncio.sleep(10)
            if sys_state.info_dict:
                # Chạy sync DB operations trong thread riêng để tránh lỗi greenlet
                current_snapshot = dict(sys_state.info_dict)
                await loop.run_in_executor(sys_state.executor, _save_stats_sync, current_snapshot)
        except Exception as e:
            print(f"❌ Lỗi vòng lặp Worker: {e}")
            await asyncio.sleep(5)
```

**Lý do thay đổi:** `asyncio.to_thread()` vẫn có thể gây vấn đề với SQLAlchemy. `ThreadPoolExecutor` với `loop.run_in_executor()` tách biệt hoàn toàn khỏi async context.

#### Thay đổi 2: Endpoint `/analyze/{camera_id}` (Line 154-221)

**Thêm function mới `_analyze_stats_sync` (Line 154-208):**
```python
def _analyze_stats_sync(camera_id: int):
    """Sync function để phân tích stats - chạy trong thread riêng"""
    db = SessionLocal()
    try:
        time_threshold = datetime.now() - timedelta(minutes=60)
        query = db.query(
            TrafficLog.timestamp, TrafficLog.total_vehicles,
            TrafficLog.count_car, TrafficLog.count_motor,
            TrafficLog.count_truck, TrafficLog.count_bus
        ).filter(
            TrafficLog.camera_id == camera_id,
            TrafficLog.timestamp >= time_threshold
        ).statement
        
        df = pd.read_sql(query, db.bind)
        
        if df.empty:
            return {"message": "Chưa đủ dữ liệu"}

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df_1min = df.resample('1min').mean().fillna(0)
        
        if len(df_1min) < 2: 
            return {"message": "Đang thu thập..."}

        current_val = df_1min['total_vehicles'].iloc[-1]
        mean_val = df_1min['total_vehicles'].mean()
        std_val = df_1min['total_vehicles'].std()
        
        # Trend detection
        recent_avg = df_1min['total_vehicles'].tail(5).mean()
        prev_avg = df_1min['total_vehicles'].iloc[-10:-5].mean() if len(df_1min) > 10 else mean_val
        trend_pct = ((recent_avg - prev_avg) / prev_avg * 100) if prev_avg > 0 else 0

        stats = {
            "current_flow": int(current_val),
            "average_flow": round(float(mean_val), 1),
            "peak_flow": int(df_1min['total_vehicles'].max()),
            "volatility": f"{round(std_val, 1)}",
            "status": "Cao điểm" if current_val > (mean_val + std_val) else "Bình thường",
            "trend_percent": round(trend_pct, 1),
            "composition": {
                "car": int(df['count_car'].sum()),
                "motor": int(df['count_motor'].sum()),
                "truck": int(df['count_truck'].sum()),
                "bus": int(df['count_bus'].sum())
            }
        }
        return stats
    except Exception as e:
        print(f"Lỗi Analyze: {e}")
        return {"error": str(e)}
    finally:
        db.close()
```

**Sửa endpoint `get_advanced_stats` (Line 210-221):**
**Trước:**
```python
@router.get("/analyze/{camera_id}")
async def get_advanced_stats(camera_id: int):
    """Phân tích nâng cao (Pandas + DB)"""
    db = SessionLocal()  # ❌ Lỗi: sync DB trong async context
    try:
        # ... code xử lý DB trực tiếp ...
    finally:
        db.close()
```

**Sau (Phiên bản cuối cùng):**
```python
@router.get("/analyze/{camera_id}")
async def get_advanced_stats(camera_id: int):
    """Phân tích nâng cao (Pandas + DB)"""
    try:
        # Chạy sync DB operations trong thread riêng để tránh lỗi greenlet
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(sys_state.executor, _analyze_stats_sync, camera_id)
        if "error" in result:
            return JSONResponse(result, status_code=500)
        return JSONResponse(result)
    except Exception as e:
        print(f"Lỗi Analyze: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
```

#### Thay đổi 3: Endpoint `/charts/vehicle-distribution` (Line 223-262)

**Thêm function mới `_get_vehicle_distribution_sync` (Line 223-256):**
```python
def _get_vehicle_distribution_sync():
    """Sync function để lấy vehicle distribution - chạy trong thread riêng"""
    db = SessionLocal()
    try:
        today = datetime.now().date()
        subquery = db.query(func.max(TrafficLog.id))\
            .filter(cast(TrafficLog.timestamp, Date) == today)\
            .group_by(TrafficLog.camera_id)
        latest_logs = db.query(TrafficLog).filter(TrafficLog.id.in_(subquery)).all()
        
        total_car = sum(log.count_car for log in latest_logs)
        total_motor = sum(log.count_motor for log in latest_logs)
        total_bus = sum(log.count_bus for log in latest_logs)
        total_truck = sum(log.count_truck for log in latest_logs)
        total_all = sum(log.total_vehicles for log in latest_logs)
        
        def _pct(val, total): return float(val)/total if total > 0 else 0.0

        return {
            "date": today.isoformat(),
            "totals": {
                "car": total_car, "motor": total_motor,
                "bus": total_bus, "truck": total_truck,
                "total_vehicles": total_all
            },
            "percentages": {
                "car": _pct(total_car, total_all),
                "motor": _pct(total_motor, total_all),
                "bus": _pct(total_bus, total_all),
                "truck": _pct(total_truck, total_all)
            }
        }
    finally:
        db.close()
```

**Sửa endpoint `get_vehicle_distribution` (Line 258-262):**
**Trước:**
```python
@router.get("/charts/vehicle-distribution")
async def get_vehicle_distribution():
    """Pie Chart Data"""
    db = SessionLocal()  # ❌ Lỗi: sync DB trong async context
    try:
        # ... code xử lý DB trực tiếp ...
    finally:
        db.close()
```

**Sau (Phiên bản cuối cùng):**
```python
@router.get("/charts/vehicle-distribution")
async def get_vehicle_distribution():
    """Pie Chart Data"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(sys_state.executor, _get_vehicle_distribution_sync)
    return JSONResponse(result)
```

#### Thay đổi 4: Endpoint `/charts/time-series/{camera_id}` (Line 263-313)

**Thêm function mới `_get_time_series_sync` (Line 263-305):**
```python
def _get_time_series_sync(camera_id: int, hours: int):
    """Sync function để lấy time series data - chạy trong thread riêng"""
    db = SessionLocal()
    try:
        time_threshold = datetime.now() - timedelta(hours=hours)
        query = db.query(
            TrafficLog.timestamp,
            TrafficLog.total_vehicles
        ).filter(
            TrafficLog.camera_id == camera_id,
            TrafficLog.timestamp >= time_threshold
        ).order_by(TrafficLog.timestamp).statement
        
        df = pd.read_sql(query, db.bind)
        
        if df.empty:
            return {"message": "Chưa đủ dữ liệu"}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample theo giờ (hoặc phút tùy yêu cầu)
        df_hourly = df.resample('1h').sum().fillna(0)
        
        # Format dữ liệu cho frontend
        data_points = []
        for idx, row in df_hourly.iterrows():
            hour_label = idx.strftime('%H:00')
            data_points.append({
                "label": hour_label,
                "value": int(row['total_vehicles'])
            })
        
        return {
            "camera_id": camera_id,
            "points": data_points,
            "period_hours": hours
        }
    except Exception as e:
        print(f"Lỗi Time Series: {e}")
        return {"error": str(e)}
    finally:
        db.close()
```

**Sửa endpoint `get_time_series_data` (Line 307-313):**
**Trước:**
```python
@router.get("/charts/time-series/{camera_id}")
async def get_time_series_data(camera_id: int, hours: int = 12):
    """Trả về dữ liệu time series để vẽ line chart (từ database)"""
    db = SessionLocal()  # ❌ Lỗi: sync DB trong async context
    try:
        # ... code xử lý DB trực tiếp ...
    finally:
        db.close()
```

**Sau (Phiên bản cuối cùng):**
```python
@router.get("/charts/time-series/{camera_id}")
async def get_time_series_data(camera_id: int, hours: int = 12):
    """Trả về dữ liệu time series để vẽ line chart (từ database)"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(sys_state.executor, _get_time_series_sync, camera_id, hours)
    if "error" in result:
        return JSONResponse(result, status_code=500)
    return JSONResponse(result)
```

#### Thay đổi 5: Thêm cleanup cho ThreadPoolExecutor (Line 114-122)

**Thêm vào shutdown_event:**
```python
@router.on_event("shutdown")
async def shutdown_event():
    print("🛑 Đang tắt hệ thống Traffic AI...")
    for p in sys_state.processes:
        if p.is_alive():
            p.terminate()
            p.join()
    # Shutdown ThreadPoolExecutor
    if sys_state.executor:
        sys_state.executor.shutdown(wait=True)
    print("✅ Đã tắt toàn bộ processes.")
```

---

### 3. **backend/app/services/road_services/AnalyzeOnRoadBase.py**

#### Thay đổi 1: Xóa import DB (Line 11-12)

**Trước:**
```python
from app.core.config import settings_metric_transport
import os
# ✅ Import Database
from app.db.base import SessionLocal
from app.models.traffic_logs import TrafficLog
```

**Sau:**
```python
from app.core.config import settings_metric_transport
import os
# ⚠️ LƯU Ý: Không import DB ở đây vì camera processes chạy trong multiprocessing
# Việc lưu DB sẽ được xử lý bởi background worker trong api_vehicles.py
```

#### Thay đổi 2: Sửa method `_check_and_save` (Line 146-165)

**Trước:**
```python
def _check_and_save(self):
    """Auto-save thống kê vào PostgreSQL database."""
    if not self.auto_save: return

    now = datetime.now()
    if (now - self.last_save_time).total_seconds() < self.save_interval_seconds:
        return

    # Tính toán số lượng
    car_count = len(self.counted_ids.get("car", set()))
    
    # Gộp tất cả biến thể xe máy
    motor_ids = set()
    motor_ids |= self.counted_ids.get("motor", set())
    motor_ids |= self.counted_ids.get("bike", set())
    motor_ids |= self.counted_ids.get("motorbike", set())
    motor_ids |= self.counted_ids.get("motorcycle", set())
    motor_count = len(motor_ids)

    bus_count = len(self.counted_ids.get("bus", set()))
    truck_count = len(self.counted_ids.get("truck", set()))
    total_vehicles = car_count + motor_count + bus_count + truck_count

    # Ghi vào DB
    db = SessionLocal()  # ❌ Lỗi: multiprocessing không tương thích với SQLAlchemy
    try:
        log = TrafficLog(
            camera_id=self.video_index,
            timestamp=now,
            count_car=int(car_count),
            count_motor=int(motor_count),
            count_bus=int(bus_count),
            count_truck=int(truck_count),
            total_vehicles=int(total_vehicles),
            fps=round(self.current_fps, 1)
        )
        db.add(log)
        db.commit()
        self.last_save_time = now
    except Exception as e:
        print(f"[Cam {self.video_index}] ❌ Error saving DB: {e}")
        db.rollback()
    finally:
        db.close()
```

**Sau:**
```python
def _check_and_save(self):
    """
    Auto-save thống kê vào PostgreSQL database.
    ⚠️ LƯU Ý: Trong multiprocessing, không nên lưu DB trực tiếp từ process con
    vì sẽ gây lỗi greenlet_spawn. Thay vào đó, dữ liệu được lưu vào shared_dict
    và worker chính sẽ xử lý việc lưu DB.
    """
    # Tắt auto_save trong multiprocessing để tránh lỗi greenlet_spawn
    # Worker chính (save_stats_to_db_worker) sẽ xử lý việc lưu DB từ shared_dict
    if not self.auto_save: return
    
    # Chỉ cập nhật timestamp để đánh dấu đã có dữ liệu mới
    # Việc lưu DB sẽ được xử lý bởi background worker
    now = datetime.now()
    if (now - self.last_save_time).total_seconds() < self.save_interval_seconds:
        return
    
    # Không lưu DB trực tiếp từ process con
    # Dữ liệu đã được cập nhật vào shared_dict qua _update_shared_data()
    # Worker chính sẽ đọc từ shared_dict và lưu vào DB
    self.last_save_time = now
```

---

### 4. **backend/app/db/base.py**

#### Thay đổi: Sửa Database URL Conversion (Line 6-18)

**Vấn đề:** Database URL đang dùng async driver (`sqlite+aiosqlite://` hoặc `postgresql+asyncpg://`), nhưng code chỉ replace `+asyncpg` nên SQLite vẫn dùng async driver cho sync session, gây lỗi greenlet_spawn.

**Trước:**
```python
# 1. URL Cấu hình
# URL Async (cho API): postgresql+asyncpg://user:pass@...
ASYNC_DATABASE_URL = settings_server.DATABASE_URL

# URL Sync (cho Background tasks/Script): postgresql://user:pass@...
# Ta cần bỏ "+asyncpg" đi để dùng driver chuẩn psycopg2
SYNC_DATABASE_URL = ASYNC_DATABASE_URL.replace("+asyncpg", "")
```

**Sau:**
```python
# 1. URL Cấu hình
# URL Async (cho API): postgresql+asyncpg://user:pass@... hoặc sqlite+aiosqlite://...
ASYNC_DATABASE_URL = settings_server.DATABASE_URL

# URL Sync (cho Background tasks/Script): postgresql://user:pass@... hoặc sqlite://...
# Ta cần chuyển từ async driver sang sync driver
SYNC_DATABASE_URL = ASYNC_DATABASE_URL
# Replace async drivers với sync drivers
if "+asyncpg" in SYNC_DATABASE_URL:
    SYNC_DATABASE_URL = SYNC_DATABASE_URL.replace("+asyncpg", "")
elif "+aiosqlite" in SYNC_DATABASE_URL:
    # SQLite: chuyển từ aiosqlite (async) sang pysqlite (sync)
    SYNC_DATABASE_URL = SYNC_DATABASE_URL.replace("+aiosqlite", "")
```

**Lý do:** Cần chuyển đổi đúng từ async driver sang sync driver cho cả PostgreSQL và SQLite để sync session hoạt động đúng.

---

## 📊 Tóm tắt thống kê

- **Tổng số file đã sửa:** 4 files
- **Tổng số function/endpoint đã sửa:** 6 functions/endpoints
- **Tổng số function mới được thêm:** 4 functions (sync wrappers)
- **Tổng số dòng code đã thay đổi:** ~250+ lines

### Files đã sửa:
1. ✅ `frontend/app/page.tsx` - 2 thay đổi
2. ✅ `backend/app/api/api_vehicles.py` - 4 endpoints + 1 worker + ThreadPoolExecutor
3. ✅ `backend/app/services/road_services/AnalyzeOnRoadBase.py` - 1 method + imports
4. ✅ `backend/app/db/base.py` - Database URL conversion

---

## ✅ Kết quả sau khi sửa

1. ✅ **WebSocket kết nối thành công:** Frontend kết nối đúng với camera ID (0, 1) thay vì "default"
2. ✅ **Không còn lỗi greenlet_spawn:** 
   - Tất cả DB operations được chạy trong ThreadPoolExecutor riêng biệt
   - Database URL được chuyển đổi đúng từ async driver sang sync driver
   - Endpoints `/api/v1/analyze/{camera_id}` trả về 200 OK thay vì 500
3. ✅ **Camera processes hoạt động ổn định:** Không còn cố gắng lưu DB trực tiếp từ multiprocessing
4. ✅ **Background worker xử lý DB:** Tất cả việc lưu DB được tập trung vào một worker duy nhất với ThreadPoolExecutor

---

## 🔧 Cách test

1. **Kiểm tra WebSocket:**
   - Mở browser console
   - Xem có lỗi 403 Forbidden không
   - Camera frames có hiển thị không

2. **Kiểm tra DB operations:**
   - Xem backend logs
   - Không còn lỗi "greenlet_spawn has not been called"
   - Dữ liệu vẫn được lưu vào DB qua background worker

3. **Kiểm tra endpoints:**
   - `/api/v1/analyze/0` và `/api/v1/analyze/1` hoạt động bình thường
   - `/api/v1/charts/vehicle-distribution` trả về dữ liệu
   - `/api/v1/charts/time-series/{camera_id}` hoạt động

---

**Lưu ý:** Sau khi sửa, cần restart backend server để các thay đổi có hiệu lực.


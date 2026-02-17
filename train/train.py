from ultralytics import YOLO
import os
import shutil

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    # 데이터셋 경로 (my_data 폴더)
    data_path = os.path.abspath('my_data/data.yaml')

    print(f"🚀 학습 시작! (Early Stopping 적용됨)")

    results = model.train(
        data=data_path,
        epochs=100,  # 최대 100번까지 시도
        imgsz=640,
        batch=16,
        plots=True,
        device='cpu',  # CPU 사용 ★★★★GPU 사용시 변경★★★★
        patience=10  # 얼리스탑 설정. 10 epoch 동안 성능 개선 없으면 강제 종료
    )

    # =========================================================
    # 모델 꺼내오기
    # =========================================================
    print("\n✅ 학습 완료! (또는 조기 종료됨)")
    print("가장 성능이 좋았던 모델(Best)을 복사합니다...")

    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    destination_path = 'my_od_model.pt'

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, destination_path)
        print(f"🎉 모델 저장 완료: {os.path.abspath(destination_path)}")
    else:
        print(f"⚠️ 파일을 찾을 수 없습니다.")
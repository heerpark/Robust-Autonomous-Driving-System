import os
import random
import shutil
from typing import List

# =======================================================
# ⚙️ 설정 (이 세 변수를 사용 환경에 맞게 수정하세요)
# =======================================================
# 1. 원본 이미지가 있는 폴더 경로 (예: 'C:\Users\User\Pictures\AllPhotos')
SOURCE_FOLDER = "/Users/username/Desktop/OriginalImages"  

# 2. 샘플링된 이미지를 복사할 새 폴더 경로 (폴더가 없으면 자동 생성됩니다)
DEST_FOLDER = "/Users/username/Desktop/Sampled_Images"   

# 3. 랜덤으로 선택할 파일 개수 (N)
N_SAMPLES = 500
# =======================================================


def get_image_files(folder_path: str) -> List[str]:
    """지정된 폴더에서 이미지 파일의 전체 경로 목록을 반환합니다."""
    # 일반적으로 사용되는 이미지 확장자
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    print(f"🔎 원본 폴더 스캔 중: {folder_path}")
    
    try:
        all_files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"❌ 오류: 원본 폴더를 찾을 수 없습니다: {folder_path}")
        return []

    # 이미지 파일만 필터링하고 전체 경로를 만듭니다.
    full_paths = []
    for f in all_files:
        if f.lower().endswith(image_extensions):
            full_paths.append(os.path.join(folder_path, f))
            
    print(f"총 {len(full_paths)}개의 이미지 파일을 찾았습니다.")
    return full_paths

def random_sample_images(file_list: List[str], n: int) -> List[str]:
    """파일 목록에서 n개만큼 랜덤으로 중복 없이 샘플링합니다."""
    total_count = len(file_list)
    
    if total_count == 0:
        print("경고: 샘플링할 이미지가 없습니다.")
        return []

    if n > total_count:
        print(f"⚠️ 경고: 요청된 샘플 개수({n})가 전체 파일 개수({total_count})보다 많습니다.")
        print("전체 파일을 복사 대상으로 선택합니다.")
        return file_list
    
    # random.sample()을 사용하여 중복 없는 n개의 샘플을 추출합니다.
    sampled_files = random.sample(file_list, n)
    
    print(f"✅ {n}개의 파일을 랜덤 샘플링했습니다.")
    return sampled_files

def copy_sampled_files(sampled_files: List[str], destination_folder: str):
    """샘플링된 파일을 대상 폴더로 복사합니다 (원본 유지)."""
    if not sampled_files:
        print("❌ 복사할 파일 목록이 비어 있습니다. 작업을 종료합니다.")
        return

    # 대상 폴더가 없으면 생성합니다.
    os.makedirs(destination_folder, exist_ok=True)
    print(f"📂 대상 폴더 준비 완료: {destination_folder}")
    
    for i, src_path in enumerate(sampled_files):
        # 파일 이름만 추출
        file_name = os.path.basename(src_path)
        # 대상 경로 생성
        dst_path = os.path.join(destination_folder, file_name)
        
        try:
            # 파일을 복사합니다 (원본 유지). copy2는 파일의 메타데이터(생성/수정 시간 등)도 함께 복사합니다.
            shutil.copy2(src_path, dst_path)
            print(f"({i+1}/{len(sampled_files)}) 복사 성공: {file_name}")
        except Exception as e:
            print(f"❌ 복사 오류 ({file_name}): {e}")

# =======================================================
# 🚀 메인 실행 부분
# =======================================================

if __name__ == "__main__":
    # 1. 파일 목록 가져오기
    all_images = get_image_files(SOURCE_FOLDER)

    # 2. n개 샘플링
    sampled_list = random_sample_images(all_images, N_SAMPLES)

    # 3. 새 폴더로 복사
    copy_sampled_files(sampled_list, DEST_FOLDER)

    print("\n🎉 모든 작업이 완료되었습니다.")
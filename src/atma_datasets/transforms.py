import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            # NOTE: HorizontalFlip は「画像＋数値を同期」させたいのでDataset内で手動でやる
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )